package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/gif"
	"image/jpeg"
	"image/png"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	totalCells = 9
	imgW       = 256
	imgH       = 192
	Port       = 8989
)

// ============================================================================
// FFT ENGINE
// ============================================================================

func fft1d(a []complex128, inverse bool) {
	n := len(a)
	if n <= 1 {
		return
	}
	for i, j := 0, 0; i < n; i++ {
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
		k := n >> 1
		for k > 0 && j >= k {
			j -= k
			k >>= 1
		}
		j += k
	}
	for length := 2; length <= n; length <<= 1 {
		angle := 2.0 * math.Pi / float64(length)
		if inverse {
			angle = -angle
		}
		wLen := complex(math.Cos(angle), math.Sin(angle))
		half := length >> 1
		for i := 0; i < n; i += length {
			w := complex(1, 0)
			for j := 0; j < half; j++ {
				u := a[i+j]
				t := w * a[i+j+half]
				a[i+j] = u + t
				a[i+j+half] = u - t
				w *= wLen
			}
		}
	}
	if inverse {
		invN := complex(1.0/float64(n), 0)
		for i := range a {
			a[i] *= invN
		}
	}
}

// fft2d transforms rows and columns, parallelized across CPU cores. Rows
// are independent slices; column workers touch disjoint columns with private
// scratch buffers, so both passes are race-free without locks.
func fft2d(data [][]complex128, inverse bool) {
	rows := len(data)
	cols := len(data[0])

	workers := runtime.NumCPU()
	if workers > rows {
		workers = rows
	}
	if workers > cols {
		workers = cols
	}
	if workers < 1 {
		workers = 1
	}

	var wg sync.WaitGroup

	// Rows.
	rchunk := (rows + workers - 1) / workers
	for w := 0; w < workers; w++ {
		lo := w * rchunk
		if lo >= rows {
			break
		}
		hi := lo + rchunk
		if hi > rows {
			hi = rows
		}
		wg.Add(1)
		go func(a, b int) {
			defer wg.Done()
			for r := a; r < b; r++ {
				fft1d(data[r], inverse)
			}
		}(lo, hi)
	}
	wg.Wait()

	// Columns: each worker gets its own scratch buffer.
	cchunk := (cols + workers - 1) / workers
	for w := 0; w < workers; w++ {
		lo := w * cchunk
		if lo >= cols {
			break
		}
		hi := lo + cchunk
		if hi > cols {
			hi = cols
		}
		wg.Add(1)
		go func(a, b int) {
			defer wg.Done()
			buf := make([]complex128, rows)
			for c := a; c < b; c++ {
				for r := 0; r < rows; r++ {
					buf[r] = data[r][c]
				}
				fft1d(buf, inverse)
				for r := 0; r < rows; r++ {
					data[r][c] = buf[r]
				}
			}
		}(lo, hi)
	}
	wg.Wait()
}

func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

func freqCoord(idx, size int) float64 {
	if idx <= size/2 {
		return float64(idx)
	}
	return float64(idx - size)
}

func minMax(data []float64) (float64, float64) {
	mn, mx := data[0], data[0]
	for _, v := range data {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	return mn, mx
}

// ============================================================================
// GENOME
// ============================================================================

type Genome struct {
	Seed          int64      `json:"seed"`
	Exponent      float64    `json:"exponent"`
	BandLimit     float64    `json:"band_limit"`
	AxisStretch   float64    `json:"axis_stretch"` // anisotropy: stretches fv before radial freq
	Gamma         float64    `json:"gamma"`        // contrast curve on t before palette
	Colorfulness  float64    `json:"colorfulness"` // palette color vs grayscale mix
	MutationRate  float64    `json:"mutation_rate"`
	MutationPower float64    `json:"mutation_power"`
	PalA          [3]float64 `json:"pal_a"` // cosine palette: base offset
	PalB          [3]float64 `json:"pal_b"` // amplitude
	PalC          [3]float64 `json:"pal_c"` // frequency
	PalD          [3]float64 `json:"pal_d"` // phase

	// Optional phase-match reference: base64 PNG of the target's normalized
	// luminance. Non-empty => phase-preserving "match" rendering mode.
	LumaRef  string  `json:"luma_ref,omitempty"`
	LumaW    int     `json:"luma_w,omitempty"`
	LumaH    int     `json:"luma_h,omitempty"`
	PhaseMix float64 `json:"phase_mix,omitempty"` // 1.0 = full target phase

	// Match-mode variation: radians of seed-driven random phase jitter.
	// 0 = exact reproduction of the reference layout; higher = wilder.
	PhaseJitter float64 `json:"phase_jitter,omitempty"`

	// Structural genes: geometric framing of the luma reference.
	// Zoom 1 = full frame; Rot in radians; Flip* are 0/1; Center in [-0.5, 0.5].
	Zoom    float64 `json:"zoom,omitempty"`
	Rot     float64 `json:"rot,omitempty"`
	FlipX   float64 `json:"flip_x,omitempty"`
	FlipY   float64 `json:"flip_y,omitempty"`
	CenterX float64 `json:"center_x,omitempty"`
	CenterY float64 `json:"center_y,omitempty"`

	// Morph gene: 1.0 = full imported-image structure, 0.0 = pure random
	// spectral synthesis. Drifts downward over generations.
	Structure float64 `json:"structure,omitempty"`

	// Warp: strength of a seed-driven smooth displacement field applied to
	// the imported structure. 0 = rigid, ~0.3 = strong liquid morphing.
	Warp float64 `json:"warp,omitempty"`

	// --- v2 visual genes (defaults reproduce the classic look) ---

	// Nonlinear transform of the normalized luminance before the palette:
	// 0 = none, 1 = turbulence (1-|2t-1|), 2 = ridged (squared turbulence),
	// 3 = terraces (soft-quantized levels).
	Transform     int     `json:"transform,omitempty"`
	TerraceLevels float64 `json:"terrace_levels,omitempty"`

	// Relief lighting: shade added from the luminance gradient.
	ReliefAngle    float64 `json:"relief_angle,omitempty"`    // light direction (radians)
	ReliefStrength float64 `json:"relief_strength,omitempty"` // 0 = off

	// Spectral breakpoint: ExponentHi replaces Exponent for frequencies
	// above BreakFreq*maxFreq. BreakFreq 0 = single-exponent (classic).
	ExponentHi float64 `json:"exponent_hi,omitempty"`
	BreakFreq  float64 `json:"break_freq,omitempty"`

	// Spectral spikes: a few seed-chosen frequency bins get their amplitude
	// multiplied by SpikeAmp -> quasi-periodic stripes / lattice motifs.
	SpikeCount int     `json:"spike_count,omitempty"`
	SpikeAmp   float64 `json:"spike_amp,omitempty"`

	// Chroma modulation: a second, independent field offsets t before the
	// palette, giving spatially-rich iridescent color variation.
	ChromaStrength float64 `json:"chroma_strength,omitempty"`

	// Spectral rotation rotates the fu/fv axes before the radial frequency
	// computation -> diagonal grain instead of axis-aligned streaks.
	SpecRot float64 `json:"spec_rot,omitempty"`

	// Directional cone: frequencies outside +/-ConeWidth*pi around ConeAngle
	// are damped -> brushed / woven looks. ConeWidth 0 or 1 = disabled.
	ConeAngle float64 `json:"cone_angle,omitempty"`
	ConeWidth float64 `json:"cone_width,omitempty"`

	// Domain warp (classic mode): smooth seed-driven displacement applied to
	// the synthesized field before normalization -> marble / flow looks.
	DomainWarp float64 `json:"domain_warp,omitempty"`

	// Normalization mode: 0 = min-max, 1 = percentile clip, 2 = rank equalize.
	NormMode int `json:"norm_mode,omitempty"`

	// Radial symmetry: SymmetryFold k (>=2) folds the luminance k-fold
	// around the image center; <2 = off. SymmetryMirror mirrors alternate
	// wedges -> kaleidoscope / mandala structures.
	SymmetryFold   int  `json:"sym_fold,omitempty"`
	SymmetryMirror bool `json:"sym_mirror,omitempty"`

	// Palette family: 0 = classic cosine palette (PalA-PalD), 1 = anchor
	// points. AnchorCount (2..5) stops from AnchorColors, lerped over t.
	PaletteMode  int           `json:"palette_mode,omitempty"`
	AnchorCount  int           `json:"anchor_count,omitempty"`
	AnchorColors [5][3]float64 `json:"anchor_colors,omitempty"`
}

type SavedGenome struct {
	Version   string `json:"version"`
	CellIndex int    `json:"cell_index"`
	Timestamp string `json:"timestamp"`
	Genome    Genome `json:"genome"`
}

// Hand-picked starting palettes (IQ-style presets) so fresh grids look
// attractive immediately instead of spending generations on the seed lottery.
type Palette struct {
	A, B, C, D [3]float64
}

var palettePresets = []Palette{
	// Rainbow
	{[3]float64{0.5, 0.5, 0.5}, [3]float64{0.5, 0.5, 0.5}, [3]float64{1, 1, 1}, [3]float64{0.0, 0.33, 0.67}},
	// Sunset
	{[3]float64{0.5, 0.5, 0.5}, [3]float64{0.5, 0.5, 0.5}, [3]float64{1, 1, 1}, [3]float64{0.10, 0.25, 0.45}},
	// Dusk (blue-orange)
	{[3]float64{0.5, 0.5, 0.5}, [3]float64{0.5, 0.5, 0.5}, [3]float64{1, 1, 1}, [3]float64{0.0, 0.10, 0.20}},
	// Yellow-pink
	{[3]float64{0.5, 0.5, 0.5}, [3]float64{0.5, 0.5, 0.5}, [3]float64{1, 1, 0.5}, [3]float64{0.8, 0.9, 0.3}},
	// Yellow-green-purple
	{[3]float64{0.5, 0.5, 0.5}, [3]float64{0.5, 0.5, 0.5}, [3]float64{2, 1, 0}, [3]float64{0.5, 0.2, 0.25}},
	// Deep ocean
	{[3]float64{0.66, 0.5, 0.5}, [3]float64{0.5, 0.4, 0.4}, [3]float64{1, 1, 0.8}, [3]float64{0.0, 0.15, 0.3}},
	// Neon
	{[3]float64{0.66, 0.5, 0.5}, [3]float64{0.5, 0.3, 0.4}, [3]float64{1, 1, 1}, [3]float64{0.0, 0.1, 0.2}},
	// Soft pastel
	{[3]float64{0.85, 0.85, 0.85}, [3]float64{0.15, 0.12, 0.1}, [3]float64{1, 1, 1}, [3]float64{0.0, 0.1, 0.2}},
}

func pickPalette(rng *rand.Rand) Palette {
	return palettePresets[rng.Intn(len(palettePresets))]
}

func randomPaletteGenes(rng *rand.Rand) (a, b, c, d [3]float64) {
	for i := 0; i < 3; i++ {
		a[i] = rng.Float64()
		b[i] = 0.3 + rng.Float64()*0.4
		c[i] = rng.Float64()
		d[i] = rng.Float64()
	}
	return
}

func randomGenome(rng *rand.Rand) Genome {
	g := Genome{
		Seed:           rng.Int63(),
		Exponent:       1.5 + rng.Float64()*2.0,
		BandLimit:      0.3 + rng.Float64()*0.6,
		AxisStretch:    0.5 + rng.Float64()*1.5,
		Gamma:          0.5 + rng.Float64()*1.5,
		Colorfulness:   0.4 + rng.Float64()*0.6,
		MutationRate:   0.001 + rng.Float64()*0.049,
		MutationPower:  1.0 + rng.Float64()*49.0,
		Transform:      rng.Intn(4),
		TerraceLevels:  3.0 + rng.Float64()*12.0,
		ReliefAngle:    rng.Float64() * 2.0 * math.Pi,
		ReliefStrength: rng.Float64() * rng.Float64() * 1.2, // skewed toward subtle
		ExponentHi:     1.5 + rng.Float64()*2.0,
		BreakFreq:      0.15 + rng.Float64()*0.5,
		SpikeCount:     rng.Intn(5), // 0..4 spikes, 0 = classic
		SpikeAmp:       2.0 + rng.Float64()*14.0,
		ChromaStrength: rng.Float64() * rng.Float64() * 0.5, // skewed toward subtle
		SpecRot:        rng.Float64() * 2.0 * math.Pi,
		ConeAngle:      rng.Float64() * 2.0 * math.Pi,
		ConeWidth:      1.0,                                  // disabled by default; rolled below
		DomainWarp:     rng.Float64() * rng.Float64() * 0.35, // skewed subtle
		NormMode:       rng.Intn(3),
	}
	// 30% of genomes start from a known-good palette
	if rng.Float64() < 0.3 {
		p := pickPalette(rng)
		g.PalA, g.PalB, g.PalC, g.PalD = p.A, p.B, p.C, p.D
	} else {
		g.PalA, g.PalB, g.PalC, g.PalD = randomPaletteGenes(rng)
	}
	// 40% of fresh genomes get a directional cone; the rest stay diffuse.
	if rng.Float64() < 0.4 {
		g.ConeWidth = 0.2 + rng.Float64()*0.6
	}
	// 30% use the anchor-point palette instead of the cosine palette.
	if rng.Float64() < 0.3 {
		g.PaletteMode = 1
		g.AnchorCount = 2 + rng.Intn(4)
		for k := 0; k < g.AnchorCount; k++ {
			for i := 0; i < 3; i++ {
				g.AnchorColors[k][i] = rng.Float64()
			}
		}
	}
	// 25% get k-fold radial symmetry (kaleidoscope / mandala structure).
	if rng.Float64() < 0.25 {
		g.SymmetryFold = 2 + rng.Intn(7)
		g.SymmetryMirror = rng.Float64() < 0.5
	}
	return g
}

func jitter3(v [3]float64, amt float64, rng *rand.Rand) [3]float64 {
	var out [3]float64
	for i := 0; i < 3; i++ {
		out[i] = clampF(v[i]+(rng.Float64()*2-1)*amt, 0.0, 2.0)
	}
	return out
}

// breedGenome creates a child genome that inherits traits from the clicked
// parent (primary, favored) and every locked cell (co-parents). Each scalar
// gene comes from one donor; the palette is taken as one coherent set; the
// seed mostly keeps the clicked parent's phase structure so children visibly
// resemble the image you clicked.
func breedGenome(parent Genome, donors []Genome, rng *rand.Rand) Genome {
	child := Genome{}

	inherit := func(get func(Genome) float64) float64 {
		if len(donors) == 0 || rng.Float64() < 0.5 {
			return get(parent)
		}
		return get(donors[rng.Intn(len(donors))])
	}

	child.Exponent = inherit(func(g Genome) float64 { return g.Exponent })
	child.BandLimit = inherit(func(g Genome) float64 { return g.BandLimit })
	child.AxisStretch = inherit(func(g Genome) float64 { return g.AxisStretch })
	child.Gamma = inherit(func(g Genome) float64 { return g.Gamma })
	child.Colorfulness = inherit(func(g Genome) float64 { return g.Colorfulness })
	child.MutationRate = inherit(func(g Genome) float64 { return g.MutationRate })
	child.MutationPower = inherit(func(g Genome) float64 { return g.MutationPower })
	child.Transform = int(inherit(func(g Genome) float64 { return float64(g.Transform) }))
	child.TerraceLevels = inherit(func(g Genome) float64 { return g.TerraceLevels })
	child.ReliefAngle = inherit(func(g Genome) float64 { return g.ReliefAngle })
	child.ReliefStrength = inherit(func(g Genome) float64 { return g.ReliefStrength })
	child.ExponentHi = inherit(func(g Genome) float64 { return g.ExponentHi })
	child.BreakFreq = inherit(func(g Genome) float64 { return g.BreakFreq })
	child.SpikeCount = int(inherit(func(g Genome) float64 { return float64(g.SpikeCount) }))
	child.SpikeAmp = inherit(func(g Genome) float64 { return g.SpikeAmp })
	child.ChromaStrength = inherit(func(g Genome) float64 { return g.ChromaStrength })
	child.SpecRot = inherit(func(g Genome) float64 { return g.SpecRot })
	child.ConeAngle = inherit(func(g Genome) float64 { return g.ConeAngle })
	child.ConeWidth = inherit(func(g Genome) float64 { return g.ConeWidth })
	child.DomainWarp = inherit(func(g Genome) float64 { return g.DomainWarp })
	child.NormMode = int(inherit(func(g Genome) float64 { return float64(g.NormMode) }))
	child.SymmetryFold = int(inherit(func(g Genome) float64 { return float64(g.SymmetryFold) }))
	child.SymmetryMirror = parent.SymmetryMirror
	if len(donors) > 0 && rng.Float64() < 0.3 {
		child.SymmetryMirror = donors[rng.Intn(len(donors))].SymmetryMirror
	}

	palFrom := parent
	if len(donors) > 0 && rng.Float64() < 0.4 {
		palFrom = donors[rng.Intn(len(donors))]
	}
	child.PalA, child.PalB = palFrom.PalA, palFrom.PalB
	child.PalC, child.PalD = palFrom.PalC, palFrom.PalD
	child.PaletteMode = palFrom.PaletteMode
	child.AnchorCount = palFrom.AnchorCount
	child.AnchorColors = palFrom.AnchorColors

	switch {
	case rng.Float64() < 0.55:
		child.Seed = parent.Seed
	case len(donors) > 0 && rng.Float64() < 0.5:
		child.Seed = donors[rng.Intn(len(donors))].Seed
	default:
		child.Seed = rng.Int63()
	}

	// --- phase reference inheritance ---
	switch {
	case rng.Float64() < 0.75:
		child.LumaRef, child.LumaW, child.LumaH = parent.LumaRef, parent.LumaW, parent.LumaH
		child.PhaseMix, child.PhaseJitter = parent.PhaseMix, parent.PhaseJitter
		child.Structure, child.Warp = parent.Structure, parent.Warp
		// structural genes: mostly from parent, sometimes recombined
		if rng.Float64() < 0.7 {
			child.Zoom, child.Rot = parent.Zoom, parent.Rot
			child.FlipX, child.FlipY = parent.FlipX, parent.FlipY
			child.CenterX, child.CenterY = parent.CenterX, parent.CenterY
		} else if len(donors) > 0 {
			d := donors[rng.Intn(len(donors))]
			child.Zoom, child.Rot = d.Zoom, d.Rot
			child.FlipX, child.FlipY = d.FlipX, d.FlipY
			child.CenterX, child.CenterY = d.CenterX, d.CenterY
		}
	case len(donors) > 0 && rng.Float64() < 0.5:
		d := donors[rng.Intn(len(donors))]
		child.LumaRef, child.LumaW, child.LumaH = d.LumaRef, d.LumaW, d.LumaH
		child.PhaseMix, child.PhaseJitter = d.PhaseMix, d.PhaseJitter
		child.Zoom, child.Rot = d.Zoom, d.Rot
		child.FlipX, child.FlipY = d.FlipX, d.FlipY
		child.CenterX, child.CenterY = d.CenterX, d.CenterY
	}

	// --- mutation ---
	jitter := func(v, amt float64) float64 {
		return v + (rng.Float64()*2-1)*amt
	}
	child.Exponent = math.Max(0.5, jitter(child.Exponent, 0.3))
	child.BandLimit = clampF(jitter(child.BandLimit, 0.1), 0.01, 1.0)
	child.AxisStretch = clampF(jitter(child.AxisStretch, 0.15), 0.25, 4.0)
	child.Gamma = clampF(jitter(child.Gamma, 0.15), 0.3, 3.0)
	child.Colorfulness = clampF(jitter(child.Colorfulness, 0.1), 0.0, 1.0)
	child.MutationRate = clampF(jitter(child.MutationRate, 0.005), 0.0001, 0.1)
	child.MutationPower = clampF(jitter(child.MutationPower, 5.0), 1.0, 100.0)
	child.ExponentHi = clampF(jitter(child.ExponentHi, 0.3), 0.5, 10.0)
	child.BreakFreq = clampF(jitter(child.BreakFreq, 0.08), 0.0, 0.9)
	child.TerraceLevels = clampF(jitter(child.TerraceLevels, 1.5), 2.0, 24.0)
	child.ReliefAngle = math.Mod(child.ReliefAngle+(rng.Float64()*2-1)*0.3+2.0*math.Pi, 2.0*math.Pi)
	child.ReliefStrength = clampF(jitter(child.ReliefStrength, 0.12), 0.0, 2.0)
	if rng.Float64() < 0.12 {
		child.Transform = rng.Intn(4)
	}
	child.SpikeAmp = clampF(jitter(child.SpikeAmp, 2.5), 0.0, 20.0)
	if rng.Float64() < 0.15 {
		child.SpikeCount = rng.Intn(5) // spike count mutates discretely
	}
	child.ChromaStrength = clampF(jitter(child.ChromaStrength, 0.08), 0.0, 0.8)
	child.SpecRot = math.Mod(jitter(child.SpecRot, 0.4)+2*math.Pi, 2*math.Pi)
	child.ConeAngle = math.Mod(jitter(child.ConeAngle, 0.4)+2*math.Pi, 2*math.Pi)
	child.ConeWidth = clampF(jitter(child.ConeWidth, 0.1), 0.0, 1.0)
	child.DomainWarp = clampF(jitter(child.DomainWarp, 0.08), 0.0, 0.5)
	if rng.Float64() < 0.1 {
		child.NormMode = rng.Intn(3)
	}
	if rng.Float64() < 0.1 {
		child.SymmetryFold = rng.Intn(9) // 0..8; <2 disables
	}
	if rng.Float64() < 0.12 {
		child.SymmetryMirror = !child.SymmetryMirror
	}
	if rng.Float64() < 0.05 { // rare: swap palette family
		if child.PaletteMode == 1 {
			child.PaletteMode = 0
		} else {
			child.PaletteMode = 1
			if child.AnchorCount < 2 {
				child.AnchorCount = 2 + rng.Intn(4)
				for k := 0; k < child.AnchorCount; k++ {
					for i := 0; i < 3; i++ {
						child.AnchorColors[k][i] = rng.Float64()
					}
				}
			}
		}
	}
	if child.PaletteMode == 1 {
		for k := 0; k < child.AnchorCount && k < 5; k++ {
			for i := 0; i < 3; i++ {
				child.AnchorColors[k][i] = clampF(child.AnchorColors[k][i]+(rng.Float64()*2-1)*0.06, 0, 1)
			}
		}
	}
	if rng.Float64() < 0.08 { // occasionally flip the cone on/off
		if child.ConeWidth > 0.999 {
			child.ConeWidth = 0.2 + rng.Float64()*0.6
		} else {
			child.ConeWidth = 1.0
		}
	}
	child.PalA = jitter3(child.PalA, 0.1, rng)
	child.PalB = jitter3(child.PalB, 0.1, rng)
	child.PalC = jitter3(child.PalC, 0.05, rng)

	if rng.Float64() < 0.7 {
		for i := 0; i < 3; i++ {
			child.PalD[i] = math.Mod(child.PalD[i]+(rng.Float64()*2-1)*0.05+1.0, 1.0)
		}
	} else {
		child.PalD = jitter3(child.PalD, 0.1, rng)
	}

	if child.LumaRef != "" {
		// Structural drift: jitter, occasional flips, gentle reframing.
		child.Structure = clampF(jitter(child.Structure, 0.12)-0.06, 0.0, 1.0)
		// Morph dynamics: siblings spread across the morph timeline — some stay
		// close to the photo, others dive deep into noise. This is what makes
		// the grid read as frames of a morph animation.
		child.Structure = clampF(jitter(child.Structure, 0.30), 0.0, 1.0)
		child.Warp = clampF(jitter(child.Warp, 0.12), 0.0, 0.5)
		child.PhaseJitter = clampF(jitter(child.PhaseJitter, 0.15), 0.0, 1.2)
		child.Zoom = clampF(jitter(child.Zoom, 0.08), 0.9, 2.2)
		child.Rot = math.Mod(child.Rot+(rng.Float64()*2-1)*0.10+math.Pi, 2*math.Pi) - math.Pi
		child.CenterX = clampF(jitter(child.CenterX, 0.04), -0.25, 0.25)
		child.CenterY = clampF(jitter(child.CenterY, 0.04), -0.25, 0.25)
		if rng.Float64() < 0.07 {
			child.FlipX = 1 - child.FlipX
		}
		if rng.Float64() < 0.07 {
			child.FlipY = 1 - child.FlipY
		}
		if rng.Float64() < 0.08 {
			child.PhaseMix = clampF(child.PhaseMix-0.1, 0.3, 1.0)
		}
	}

	// Occasional larger reset of one gene group
	if rng.Float64() < 0.15 {
		switch rng.Intn(8) {
		case 0:
			child.Exponent = 1.5 + rng.Float64()*2.0
		case 1:
			child.BandLimit = 0.2 + rng.Float64()*0.7
		case 2:
			child.AxisStretch = 0.5 + rng.Float64()*2.0
		case 3:
			child.Colorfulness = rng.Float64()
		case 4:
			p := pickPalette(rng)
			child.PalA, child.PalB, child.PalC, child.PalD = p.A, p.B, p.C, p.D
		case 5:
			child.MutationRate = 0.001 + rng.Float64()*0.049
		case 6:
			child.PalA, child.PalB, child.PalC, child.PalD = randomPaletteGenes(rng)
		case 7:
			if child.LumaRef != "" { // rare: strong step toward randomness
				child.Structure = rng.Float64() * 0.5
			}
		}
	}
	return child
}

func clampF(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// ============================================================================
// COSINE PALETTE
// ============================================================================

// IQ-style cosine palette: color(t) = a + b * cos(2π * (c * t + d))
// t in [0, 1] -> RGB in [0, 255]
func cosinePalette(t float64, cfg Genome) (uint8, uint8, uint8) {
	var out [3]uint8
	for i := 0; i < 3; i++ {
		v := cfg.PalA[i] + cfg.PalB[i]*math.Cos(2*math.Pi*(cfg.PalC[i]*t+cfg.PalD[i]))
		out[i] = uint8(clampF(v, 0.0, 1.0) * 255.0)
	}
	return out[0], out[1], out[2]
}

// ============================================================================
// IMAGE GENERATION — DIRECT SPECTRAL SYNTHESIS
// ============================================================================

// paletteLookup dispatches between the two palette families.
func paletteLookup(t float64, cfg Genome) (uint8, uint8, uint8) {
	if cfg.PaletteMode == 1 {
		return anchorPalette(t, cfg)
	}
	return cosinePalette(t, cfg)
}

// anchorPalette lerps through the genome's anchor colors over t in [0, 1].
// The anchor family reaches subdued multi-stop ramps (film-curve pastels,
// duotones, branded ramps) the 4-vector cosine palette cannot hit.
func anchorPalette(t float64, cfg Genome) (uint8, uint8, uint8) {
	n := cfg.AnchorCount
	if n < 2 {
		n = 2
	}
	if n > 5 {
		n = 5
	}
	pos := clampF(t, 0, 1) * float64(n-1)
	k := int(pos)
	if k >= n-1 {
		k = n - 2
	}
	f := pos - float64(k)
	var out [3]uint8
	for i := 0; i < 3; i++ {
		v := cfg.AnchorColors[k][i]*(1-f) + cfg.AnchorColors[k+1][i]*f
		out[i] = uint8(clampF(v, 0, 1) * 255.0)
	}
	return out[0], out[1], out[2]
}

// smoothstep01 is the classic smooth Hermite step on [0, 1].
func smoothstep01(t float64) float64 {
	if t < 0 {
		t = 0
	}
	if t > 1 {
		t = 1
	}
	return t * t * (3.0 - 2.0*t)
}

// applyTransform reshapes the normalized luminance t in [0, 1] according to
// the genome's Transform gene. These nonlinearities break the Gaussian
// cloudiness of raw spectral noise: turbulence carves sharp dark ridges,
// ridged concentrates them into bright filaments, terraces quantize the
// field into elevation bands.
func applyTransform(t float64, cfg Genome) float64 {
	switch cfg.Transform {
	case 1: // turbulence
		return 1.0 - math.Abs(2.0*t-1.0)
	case 2: // ridged
		v := 1.0 - math.Abs(2.0*t-1.0)
		return v * v
	case 3: // terraces
		levels := cfg.TerraceLevels
		if levels < 2 {
			levels = 2
		}
		scaled := clampF(t, 0, 1) * (levels - 1.0)
		base := math.Floor(scaled)
		frac := scaled - base
		// narrow smooth transition between bands keeps edges soft but defined
		edge := smoothstep01((frac - 0.5) / 0.2)
		return (base + edge) / (levels - 1.0)
	}
	return t
}

// synthChannel generates a single noise field by shaping random complex
// coefficients directly in the frequency domain, then doing ONE inverse FFT.
// (The old path did: white noise -> forward FFT -> shape -> inverse FFT.
// Since white noise already has a flat spectrum, the forward transform was
// wasted work.)
func synthChannel(padW, padH int, rng *rand.Rand, cfg Genome) []float64 {
	data := make([][]complex128, padH)
	for y := range data {
		data[y] = make([]complex128, padW)
	}
	halfW := padW / 2
	stretch := cfg.AxisStretch

	// Spectral spikes: deterministic from Seed. Both the chosen bin and its
	// Hermitian conjugate are boosted so the real part of the inverse field
	// carries the full periodic energy.
	spikeMap := make(map[[2]int]float64)
	if cfg.SpikeCount > 0 && cfg.SpikeAmp > 0.001 {
		sr := rand.New(rand.NewSource(cfg.Seed + 31337))
		for i := 0; i < cfg.SpikeCount; i++ {
			ix := 1 + sr.Intn(halfW-1)
			iy := sr.Intn(padH) - padH/2
			wx := ((ix % padW) + padW) % padW
			wy := ((iy % padH) + padH) % padH
			spikeMap[[2]int{wx, wy}] = cfg.SpikeAmp
			spikeMap[[2]int{(padW - wx) % padW, (padH - wy) % padH}] = cfg.SpikeAmp
		}
	}

	// Spectral rotation + directional cone (precomputed factors).
	rot := cfg.SpecRot
	cr, ci := math.Cos(rot), math.Sin(rot)
	coneOn := cfg.ConeWidth > 0.001 && cfg.ConeWidth < 0.999
	halfCone := cfg.ConeWidth * math.Pi

	// Hermitian fill: randomize only the top half of the frequency plane and
	// mirror conjugates into the bottom half. Halves the RNG work and makes
	// the field exactly real by construction (no imaginary part discarded).
	// Cone attenuation is inherited by the mirrored bin, so the cone is
	// effectively two-lobed (theta and theta+pi) — which is what produces
	// the streaky brushed/fabric look rather than a one-sided flow.
	for y := 0; y <= padH/2; y++ {
		ym := (padH - y) % padH
		selfRow := ym == y // rows 0 and padH/2 pair with themselves
		xmax := padW - 1
		if selfRow {
			xmax = halfW
		}
		for x := 0; x <= xmax; x++ {
			xm := (padW - x) % padW
			fu := freqCoord(x, padW)
			fv := freqCoord(y, padH)
			if rot != 0 {
				fu, fv = fu*cr-fv*ci, fu*ci+fv*cr
			}
			fvv := fv * stretch // anisotropy: directional frequency scaling
			f := math.Sqrt(fu*fu + fvv*fvv)
			if f < 0.5 {
				continue // kill DC and the lowest bin
			}
			// Spectral breakpoint: past BreakFreq*maxFreq switch to the
			// high-frequency exponent. This decouples large-scale billows
			// from fine-grain roughness.
			e := cfg.Exponent
			if cfg.BreakFreq > 0.001 {
				corner := float64(halfW) * cfg.BreakFreq
				if f > corner {
					e = cfg.ExponentHi
				}
			}
			amp := 1.0 / math.Pow(f, e/2.0)
			if boost, ok := spikeMap[[2]int{x, y}]; ok {
				amp *= boost
			}
			if cfg.BandLimit > 0 {
				maxFreq := float64(halfW)
				cutoff := maxFreq * cfg.BandLimit
				if f > cutoff {
					falloff := math.Exp(-((f - cutoff) * (f - cutoff)) / (2 * cutoff * cutoff))
					amp *= falloff
				}
			}
			if coneOn {
				ang := math.Atan2(fvv, fu)
				d := math.Abs(math.Mod(ang-cfg.ConeAngle+3*math.Pi, 2*math.Pi) - math.Pi)
				if d > halfCone {
					amp *= math.Exp(-3.0 * (d - halfCone))
				}
			}
			if selfRow && xm == x {
				// Self-conjugate bins (DC/Nyquist corners) must be real.
				data[y][x] = complex(amp, 0)
				continue
			}
			phase := rng.Float64() * 2.0 * math.Pi
			v := complex(amp*math.Cos(phase), amp*math.Sin(phase))
			data[y][x] = v
			data[ym][xm] = complex(real(v), -imag(v))
		}
	}

	fft2d(data, true)
	result := make([]float64, padW*padH)
	for y := 0; y < padH; y++ {
		for x := 0; x < padW; x++ {
			result[y*padW+x] = real(data[y][x])
		}
	}
	return result
}

// applySymmetry folds the luminance field k-fold around the image center by
// polar remapping: the angle is reduced modulo the sector and, with mirror
// enabled, alternate wedges are reflected. Identity when SymmetryFold < 2.
func applySymmetry(lum []float64, width, height, padW int, cfg Genome) []float64 {
	k := cfg.SymmetryFold
	if k < 2 {
		return lum
	}
	out := make([]float64, len(lum))
	cx := float64(width) / 2.0
	cy := float64(height) / 2.0
	sec := 2.0 * math.Pi / float64(k)
	sample := func(x, y float64) float64 {
		xx := clampF(x, 0, float64(width-1))
		yy := clampF(y, 0, float64(height-1))
		x0, y0 := int(xx), int(yy)
		tx, ty := xx-float64(x0), yy-float64(y0)
		x1, y1 := minInt(x0+1, width-1), minInt(y0+1, height-1)
		top := lum[y0*padW+x0]*(1-tx) + lum[y0*padW+x1]*tx
		bot := lum[y1*padW+x0]*(1-tx) + lum[y1*padW+x1]*tx
		return top*(1-ty) + ty*bot
	}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			dx := float64(x) - cx
			dy := float64(y) - cy
			r := math.Hypot(dx, dy)
			a := math.Atan2(dy, dx)
			if a < 0 {
				a += 2 * math.Pi
			}
			w := math.Mod(a, sec)
			if cfg.SymmetryMirror && int(a/sec)%2 == 1 {
				w = sec - w
			}
			sx := cx + math.Cos(w)*r
			sy := cy + math.Sin(w)*r
			out[y*padW+x] = sample(sx, sy)
		}
	}
	return out
}

func generateSpectralImage(width, height int, rng *rand.Rand, cfg Genome) *image.RGBA {
	padW := nextPow2(width)
	padH := nextPow2(height)

	var luminance []float64
	if cfg.LumaRef != "" {
		structured := synthMatchChannel(width, height, cfg)
		if structured != nil {
			s := clampF(cfg.Structure, 0, 1)
			if s >= 0.999 {
				luminance = structured
			} else {
				random := synthChannel(padW, padH, rng, cfg)
				// Patchy morph mask: a smooth per-genome noise field shifts the
				// blend weight around its mean, so the dissolve sweeps across
				// the image in patches instead of fading uniformly.
				mr := rand.New(rand.NewSource(cfg.Seed + 4242))
				var mFx, mFy, mPh [3]float64
				for h := 0; h < 3; h++ {
					mFx[h] = 1.0 + mr.Float64()*3.0
					mFy[h] = 1.0 + mr.Float64()*3.0
					mPh[h] = mr.Float64() * 2 * math.Pi
				}
				for y := 0; y < height; y++ {
					uy := float64(y) / float64(height)
					for x := 0; x < width; x++ {
						ux := float64(x) / float64(width)
						d := 0.0
						for h := 0; h < 3; h++ {
							d += math.Sin(2*math.Pi*(mFx[h]*ux+mFy[h]*uy) + mPh[h])
						}
						w := clampF(s+0.30*(d/3.0), 0.0, 1.0)
						i := y*padW + x
						structured[i] = structured[i]*w + random[i]*(1.0-w)
					}
				}
				luminance = structured
			}
		}
	}
	if luminance == nil { // classic mode or broken ref
		// Universal-tile synthesis: the field is realized ONCE on a fixed
		// periodic grid defined purely by the genome, then resampled to the
		// requested canvas. Every resolution samples the same realization,
		// so exports are exactly scaled versions of the grid previews (and
		// symmetric/kaleidoscope structures keep their seams aligned).
		padT := nextPow2(fieldTile)
		canonical := synthChannel(padT, padT, rand.New(rand.NewSource(cfg.Seed)), cfg)
		luminance = resampleTile(canonical, fieldTile, padT, width, height, padW, padH)
	}

	// Domain warp (classic mode only): displace the field through a smooth
	// seed-driven flow before normalization -> marble / wood-grain looks.
	if cfg.DomainWarp > 0.001 {
		luminance = warpDomain(luminance, width, height, padW, cfg)
	}

	// Radial symmetry: fold the luminance k-fold around the center
	// (kaleidoscope / mandala). Applies to both classic and match modes.
	if cfg.SymmetryFold >= 2 {
		luminance = applySymmetry(luminance, width, height, padW, cfg)
	}

	normLum := normalize255(luminance, cfg.NormMode, width, height, padW)

	// Relief shading resolution normalization: adjacent pixels sample the
	// field at intervals proportional to wpp, so the per-pixel gradient
	// weakens on large canvases. Scale it back to the grid preview's world
	// slope (long side 256 reference) so relief looks identical everywhere.
	maxSide := width
	if height > maxSide {
		maxSide = height
	}
	reliefGain := float64(maxSide) / 256.0
	reliefSpan := int(reliefGain + 0.5)
	if reliefSpan < 1 {
		reliefSpan = 1
	}

	// Chroma modulation field: an independent spectral realization mapped to
	// [-1, 1]. Added to the palette coordinate t per pixel, it produces
	// spatially structured color shifts (iridescence) the static cosine
	// palette cannot express.
	var chromaField []float64
	if cfg.ChromaStrength > 0.001 {
		chCfg := cfg
		chCfg.Transform = 0
		chCfg.BreakFreq = 0
		chCfg.ReliefStrength = 0
		chCfg.MutationRate = 0
		chCfg.MutationPower = 0
		chPadT := nextPow2(fieldTile)
		chromaRaw := synthChannel(chPadT, chPadT, rand.New(rand.NewSource(cfg.Seed+9091)), chCfg)
		cmn, cmx := minMax(chromaRaw)
		crange := cmx - cmn
		if crange < 1e-10 {
			crange = 1
		}
		chromaField = resampleTile(chromaRaw, fieldTile, chPadT, width, height, padW, padH)
		for i := range chromaField {
			chromaField[i] = (chromaField[i]-cmn)/crange*2.0 - 1.0
		}
	}

	rgba := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			idx := y*padW + x

			lum01 := applyTransform(float64(normLum[idx])/255.0, cfg)
			if cfg.ReliefStrength > 0.001 && x > reliefSpan && x < width-1-reliefSpan && y > reliefSpan && y < height-reliefSpan {
				// Gradient measured over reliefSpan pixels keeps the world-space
				// sampling distance constant across resolutions (matches the
				// 256px preview's footprint) and suppresses byte-quantization
				// noise that reliefGain would otherwise amplify into grain.
				dxl := (float64(normLum[idx+reliefSpan]) - float64(normLum[idx-reliefSpan])) / (510.0 * float64(reliefSpan))
				dyl := (float64(normLum[idx+reliefSpan*padW]) - float64(normLum[idx-reliefSpan*padW])) / (510.0 * float64(reliefSpan))
				lx := math.Cos(cfg.ReliefAngle)
				ly := math.Sin(cfg.ReliefAngle)
				lum01 = clampF(lum01+cfg.ReliefStrength*reliefGain*(dxl*lx+dyl*ly), 0, 1)
			}
			t := math.Pow(lum01, cfg.Gamma)
			if chromaField != nil {
				t = clampF(t+cfg.ChromaStrength*chromaField[idx], 0, 1)
			}
			r, g, b := paletteLookup(t, cfg)

			gray := normLum[idx]
			cf := cfg.Colorfulness
			r = clampByte(int(float64(gray)*(1.0-cf) + float64(r)*cf))
			g = clampByte(int(float64(gray)*(1.0-cf) + float64(g)*cf))
			b = clampByte(int(float64(gray)*(1.0-cf) + float64(b)*cf))

			if rng.Float64() < cfg.MutationRate {
				mutation := int(randNorm(0, cfg.MutationPower, rng))
				r = clampByte(int(r) + mutation)
				g = clampByte(int(g) + mutation)
				b = clampByte(int(b) + mutation)
			}
			rgba.Set(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}
	return rgba
}

// fieldTile is the side of the universal synthesis tile. The classic field
// is ALWAYS realized on this fixed periodic grid — for previews, exports,
// and animation alike — then resampled to the canvas. This is what makes
// the saved image an exact scaled version of the grid cell. Raise it for
// crisper large exports; lower it if evolution renders feel slow.
const fieldTile = 1024

// resampleTile maps the fixed periodic tile onto a canvas ISOTROPICALLY:
// one uniform world-per-pixel scale anchored to the canvas long side, so
// canvases of different aspect ratios sample the same geometry with the
// same stretch (circles stay circles; symmetry seams stay put). The tile
// wraps periodically beyond its bounds, matching the FFT field. When the
// canvas samples the tile coarsely (downscaling), the footprint is
// supersampled into a low-pass average to prevent aliasing; when sampling
// finely (upscaling), a single bilinear tap gives smooth interpolation.
func resampleTile(src []float64, tile, srcStride, dw, dh, dstStride, dstRows int) []float64 {
	out := make([]float64, dstStride*dstRows)
	long := dw
	if dh > long {
		long = dh
	}
	wpp := float64(tile) / float64(long) // tile units per canvas pixel (isotropic)
	ft := float64(tile)

	samp := func(u, v float64) float64 {
		u = math.Mod(u, ft)
		if u < 0 {
			u += ft
		}
		v = math.Mod(v, ft)
		if v < 0 {
			v += ft
		}
		iu, iv := int(u), int(v)
		fu, fv := u-float64(iu), v-float64(iv)
		ju, jv := (iu+1)%tile, (iv+1)%tile
		a := src[iv*srcStride+iu]
		b := src[iv*srcStride+ju]
		c := src[jv*srcStride+iu]
		d := src[jv*srcStride+ju]
		return (a*(1-fu)+b*fu)*(1-fv) + (c*(1-fu)+d*fu)*fv
	}

	ns := int(wpp) + 1 // supersampling taps per axis (>=1)
	for y := 0; y < dstRows; y++ {
		ly := y % dh
		oy := ft/2 + (float64(ly)+0.5-float64(dh)/2)*wpp
		for x := 0; x < dstStride; x++ {
			lx := x % dw
			ox := ft/2 + (float64(lx)+0.5-float64(dw)/2)*wpp
			var acc float64
			if ns == 1 {
				acc = samp(ox, oy)
			} else {
				step := wpp / float64(ns)
				var s float64
				for ky := 0; ky < ns; ky++ {
					vv := oy - wpp/2 + (float64(ky)+0.5)*step
					for kx := 0; kx < ns; kx++ {
						uu := ox - wpp/2 + (float64(kx)+0.5)*step
						s += samp(uu, vv)
					}
				}
				acc = s / float64(ns*ns)
			}
			out[y*dstStride+x] = acc
		}
	}
	return out
}

// normalize255 maps the valid width×height sub-region (row stride padW)
// of the field to bytes. Only valid pixels contribute to the statistics,
// so results are identical at every canvas size regardless of how much
// power-of-two padding the buffer carries. This fixes exports differing
// from previews for genomes that leave the padding zeroed (DomainWarp,
// SymmetryFold, match-mode LumaRef).
func normalize255(data []float64, mode int, width, height, padW int) []byte {
	result := make([]byte, len(data))
	if width <= 0 || height <= 0 {
		return result
	}
	// Fast path: no padding at all — normalize the buffer directly.
	if width == padW && height*padW == len(data) {
		return normalizeRegion(data, mode)
	}

	// Extract the valid region as a compact contiguous copy so the
	// statistics (min/max, percentiles, ranks) see only real pixels.
	valid := make([]float64, width*height)
	for y := 0; y < height; y++ {
		copy(valid[y*width:(y+1)*width], data[y*padW:y*padW+width])
	}
	norm := normalizeRegion(valid, mode)

	// Scatter back into a padded output buffer. Padding stays zero, which
	// is safe: the render loop only indexes y*padW + x with x < width and
	// y < height, so the padding bytes are never read afterwards.
	for y := 0; y < height; y++ {
		copy(result[y*padW:y*padW+width], norm[y*width:(y+1)*width])
	}
	return result
}

// normalizeRegion maps a compact (unpadded) float field to bytes.
// mode selects the mapping:
// 0 = min-max stretch (classic), 1 = 1st/99th percentile clip,
// 2 = rank equalization (full histogram flattening).
// This is the original normalize255 body, unchanged, operating on a
// contiguous slice whose row stride equals the data width.
func normalizeRegion(data []float64, mode int) []byte {
	result := make([]byte, len(data))
	if len(data) == 0 {
		return result
	}
	if mode == 2 {
		idx := make([]int, len(data))
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return data[idx[a]] < data[idx[b]] })
		den := float64(len(data) - 1)
		if den < 1 {
			den = 1
		}
		for rank, i := range idx {
			result[i] = byte(float64(rank) / den * 255.0)
		}
		return result
	}
	lo, hi := minMax(data)
	if mode == 1 {
		sorted := append([]float64(nil), data...)
		sort.Float64s(sorted)
		lo = sorted[int(float64(len(sorted)-1)*0.01)]
		hi = sorted[int(float64(len(sorted)-1)*0.99)]
	}
	rangeVal := hi - lo
	if rangeVal < 1e-10 {
		rangeVal = 1
	}
	for i, v := range data {
		normalized := (v - lo) / rangeVal * 255.0
		if normalized < 0 {
			normalized = 0
		}
		if normalized > 255 {
			normalized = 255
		}
		result[i] = byte(normalized)
	}
	return result
}

// warpDomain displaces the classic-mode luminance field through a smooth,
// seed-driven displacement field (identity at DomainWarp 0), producing
// marbled / flowing structures that raw spectral noise cannot reach.
// Displacement is a fraction of the frame, sampled bilinearly with clamped
// edges.
func warpDomain(lum []float64, width, height, padW int, cfg Genome) []float64 {
	warp := clampF(cfg.DomainWarp, 0, 0.5)
	wr := rand.New(rand.NewSource(cfg.Seed + 5150))
	var fx, fy, ph [3]float64
	for h := 0; h < 3; h++ {
		fx[h] = 1.0 + wr.Float64()*3.0
		fy[h] = 1.0 + wr.Float64()*3.0
		ph[h] = wr.Float64() * 2 * math.Pi
	}
	sample := func(x, y float64) float64 {
		xx := clampF(x, 0, float64(width-1))
		yy := clampF(y, 0, float64(height-1))
		x0, y0 := int(xx), int(yy)
		tx, ty := xx-float64(x0), yy-float64(y0)
		x1, y1 := minInt(x0+1, width-1), minInt(y0+1, height-1)
		top := lum[y0*padW+x0]*(1-tx) + lum[y0*padW+x1]*tx
		bot := lum[y1*padW+x0]*(1-tx) + lum[y1*padW+x1]*tx
		return top*(1-ty) + ty*bot
	}
	out := make([]float64, len(lum))
	for y := 0; y < height; y++ {
		uy := float64(y) / float64(height)
		for x := 0; x < width; x++ {
			ux := float64(x) / float64(width)
			wd, wq := 0.0, 0.0
			for h := 0; h < 3; h++ {
				wd += math.Sin(2*math.Pi*(fx[h]*ux+fy[h]*uy) + ph[h])
				wq += math.Sin(2*math.Pi*(fy[h]*ux+fx[h]*uy) + ph[(h+1)%3])
			}
			ox := warp * (wd / 3.0) * float64(width)
			oy := warp * (wq / 3.0) * float64(height)
			out[y*padW+x] = sample(float64(x)+ox, float64(y)+oy)
		}
	}
	return out
}

func clampByte(v int) byte {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return byte(v)
}

func randNorm(mean, stddev float64, rng *rand.Rand) float64 {
	u1 := rng.Float64()
	u2 := rng.Float64()
	z := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
	return mean + stddev*z
}

// renderClean renders a genome deterministically with the stochastic
// per-pixel mutation noise disabled. Grid previews and exports both go
// through this, so any size renders the same pixels; only interpolation
// detail differs between resolutions. The genome is copied, never modified.
func renderClean(g Genome, width, height int) *image.RGBA {
	rc := g
	rc.MutationRate = 0
	rc.MutationPower = 0
	return generateSpectralImage(width, height, rand.New(rand.NewSource(rc.Seed)), rc)
}

// renderPreviewFramed renders a genome at the grid preview's field of
// view, supersampled 2x for a smooth, anti-aliased export. The tile is
// sampled isotropically anchored to the canvas long side, so any
// canvas narrower than 4:3 is first narrowed (keeping height) to match
// the 256x192 preview's world window. Rendering that canvas at 2x
// covers the IDENTICAL world window (wpp halves as the canvas doubles),
// so the result can be Lanczos-downsampled back to the requested size —
// suppressing the fine grain a near-1:1 display of the 1024-tile
// otherwise shows. Supersampling is skipped when the doubled canvas
// would exceed the 8192-px render budget (memory), falling back to the
// plain render: at those sizes the field buffer alone is ~0.5 GB.
func renderPreviewFramed(g Genome, width, height int) *image.RGBA {
	if maxW := height * 4 / 3; width > maxW {
		width = maxW
	}
	ss := 2
	if width*ss > 8192 || height*ss > 8192 {
		ss = 1
	}
	if ss == 1 {
		return renderClean(g, width, height)
	}
	big := renderClean(g, width*ss, height*ss)
	return downsampleImage(big, width, height)
}

// cropToPreviewFOV centrally trims a render to the 4:3 field of view the
// grid preview shows (the synthesis tile is sampled isotropically, so
// wider canvases would otherwise capture a wider slice of the field).
// Keeps full height. 1920x1080 -> 1440x1080, 7680x4320 -> 5760x4320.
func cropToPreviewFOV(img *image.RGBA) *image.RGBA {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	target := w
	if max43 := h * 4 / 3; w > max43 {
		target = max43
	}
	if target >= w {
		return img
	}
	x0 := (w - target) / 2
	out := image.NewRGBA(image.Rect(0, 0, target, h))
	draw.Draw(out, out.Bounds(), img, image.Point{X: x0, Y: 0}, draw.Src)
	return out
}

func imgToBase64(img *image.RGBA) string {
	var buf bytes.Buffer
	png.Encode(&buf, img)
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

// ============================================================================
// IMAGE UTILITIES (for reverse engineering)
// ============================================================================

// lanczos2Kernel is the 2-lobe Lanczos window: sharp without the ringing
// of 3-lobe, far better than box averaging for spectrum analysis input.
func lanczos2Kernel(x float64) float64 {
	x = math.Abs(x)
	if x < 1e-8 {
		return 1
	}
	if x >= 2 {
		return 0
	}
	px := math.Pi * x
	return 2 * math.Sin(px) * math.Sin(px/2) / (px * px)
}

func clampi(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// downsampleImage uses Lanczos-2 for strong shrinks (sharper comparison
// targets improve exponent fits and match references) and box averaging
// otherwise.
func downsampleImage(src image.Image, targetW, targetH int) *image.RGBA {
	if src.Bounds().Dx() >= 2*targetW && src.Bounds().Dy() >= 2*targetH {
		return downsampleLanczos2(src, targetW, targetH)
	}
	return downsampleBox(src, targetW, targetH)
}

// downsampleLanczos2 is a separable Lanczos-2 resize.
func downsampleLanczos2(src image.Image, targetW, targetH int) *image.RGBA {
	srcB := src.Bounds()
	srcW := srcB.Dx()
	srcH := srcB.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, targetW, targetH))
	if srcW < 1 || srcH < 1 || targetW < 1 || targetH < 1 {
		return dst
	}
	if srcW == targetW && srcH == targetH {
		for y := 0; y < targetH; y++ {
			for x := 0; x < targetW; x++ {
				r, g, b, _ := src.At(srcB.Min.X+x, srcB.Min.Y+y).RGBA()
				dst.SetRGBA(x, y, color.RGBA{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8), 255})
			}
		}
		return dst
	}

	xscale := float64(srcW) / float64(targetW)
	yscale := float64(srcH) / float64(targetH)
	xf := math.Max(1.0, xscale)
	yf := math.Max(1.0, yscale)

	// Horizontal pass: srcW -> targetW, height unchanged.
	tmp := make([]float64, srcH*targetW*3)
	for y := 0; y < srcH; y++ {
		for x := 0; x < targetW; x++ {
			center := (float64(x)+0.5)*xscale - 0.5
			lo := int(math.Ceil(center - 2.0*xf - 0.5))
			hi := int(math.Floor(center + 2.0*xf - 0.5))
			var wsum, rs, gs, bs float64
			for i := lo; i <= hi; i++ {
				d := (float64(i) + 0.5 - center - 0.5) / xf
				w := lanczos2Kernel(d)
				ii := clampi(i, 0, srcW-1)
				r, g, b, _ := src.At(srcB.Min.X+ii, srcB.Min.Y+y).RGBA()
				rs += w * float64(r>>8)
				gs += w * float64(g>>8)
				bs += w * float64(b>>8)
				wsum += w
			}
			if wsum < 1e-9 {
				wsum = 1
			}
			o := (y*targetW + x) * 3
			tmp[o] = rs / wsum
			tmp[o+1] = gs / wsum
			tmp[o+2] = bs / wsum
		}
	}

	// Vertical pass: srcH -> targetH.
	for y := 0; y < targetH; y++ {
		center := (float64(y)+0.5)*yscale - 0.5
		lo := int(math.Ceil(center - 2.0*yf - 0.5))
		hi := int(math.Floor(center + 2.0*yf - 0.5))
		for x := 0; x < targetW; x++ {
			var wsum, rs, gs, bs float64
			for i := lo; i <= hi; i++ {
				d := (float64(i) + 0.5 - center - 0.5) / yf
				w := lanczos2Kernel(d)
				ii := clampi(i, 0, srcH-1)
				o := (ii*targetW + x) * 3
				rs += w * tmp[o]
				gs += w * tmp[o+1]
				bs += w * tmp[o+2]
				wsum += w
			}
			if wsum < 1e-9 {
				wsum = 1
			}
			dst.SetRGBA(x, y, color.RGBA{
				R: uint8(clampF(rs/wsum, 0, 255)),
				G: uint8(clampF(gs/wsum, 0, 255)),
				B: uint8(clampF(bs/wsum, 0, 255)),
				A: 255,
			})
		}
	}
	return dst
}

// downsampleBox is the original box-average resize, kept as the gentle-
// scale path.
func downsampleBox(src image.Image, targetW, targetH int) *image.RGBA {
	srcB := src.Bounds()
	srcW := srcB.Dx()
	srcH := srcB.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, targetW, targetH))
	for y := 0; y < targetH; y++ {
		for x := 0; x < targetW; x++ {
			sx0 := srcB.Min.X + (x * srcW / targetW)
			sy0 := srcB.Min.Y + (y * srcH / targetH)
			sx1 := srcB.Min.X + ((x + 1) * srcW / targetW)
			sy1 := srcB.Min.Y + ((y + 1) * srcH / targetH)
			if sx1 <= sx0 {
				sx1 = sx0 + 1
			}
			if sy1 <= sy0 {
				sy1 = sy0 + 1
			}
			var rSum, gSum, bSum, count uint64
			for sy := sy0; sy < sy1; sy++ {
				for sx := sx0; sx < sx1; sx++ {
					r, g, b, _ := src.At(sx, sy).RGBA()
					rSum += uint64(r >> 8)
					gSum += uint64(g >> 8)
					bSum += uint64(b >> 8)
					count++
				}
			}
			if count == 0 {
				count = 1
			}
			dst.SetRGBA(x, y, color.RGBA{
				R: uint8(rSum / count),
				G: uint8(gSum / count),
				B: uint8(bSum / count),
				A: 255,
			})
		}
	}
	return dst
}

func compareImages(a, b *image.RGBA) float64 {
	bounds := a.Bounds()
	if b.Bounds().Dx() != bounds.Dx() || b.Bounds().Dy() != bounds.Dy() {
		return math.MaxFloat64
	}
	var sumSqDiff float64
	var count float64
	for y := 0; y < bounds.Dy(); y++ {
		for x := 0; x < bounds.Dx(); x++ {
			ar, ag, ab, _ := a.At(x, y).RGBA()
			br, bg, bb, _ := b.At(x, y).RGBA()
			dr := float64(ar>>8) - float64(br>>8)
			dg := float64(ag>>8) - float64(bg>>8)
			db := float64(ab>>8) - float64(bb>>8)
			sumSqDiff += dr*dr + dg*dg + db*db
			count += 3
		}
	}
	return sumSqDiff / count
}

// ============================================================================
// STRUCTURAL ANALYSIS (for reverse engineering)
// ============================================================================

// estimateExponent regresses log(power) vs log(freq) on the image's power
// spectrum — used both for initial estimation and structural scoring.
func estimateExponent(img *image.RGBA) float64 {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	padW := nextPow2(w)
	padH := nextPow2(h)
	grayData := make([][]complex128, padH)
	for y := 0; y < padH; y++ {
		grayData[y] = make([]complex128, padW)
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			gray := float64(r+g+b) / 3.0 / 65535.0
			grayData[y][x] = complex(gray*2-1, 0)
		}
	}
	fft2d(grayData, false)
	halfW := padW / 2
	powerSum := make([]float64, halfW+1)
	powerCount := make([]int, halfW+1)
	for y := 0; y < padH; y++ {
		for x := 0; x < padW; x++ {
			fu := freqCoord(x, padW)
			fv := freqCoord(y, padH)
			r := int(math.Sqrt(fu*fu + fv*fv))
			if r >= 0 && r <= halfW {
				re := real(grayData[y][x])
				im := imag(grayData[y][x])
				powerSum[r] += re*re + im*im
				powerCount[r]++
			}
		}
	}
	var sumLF, sumLP, sumLF2, sumLFLP float64
	nPts := 0
	for r := 2; r < halfW; r++ {
		if powerCount[r] > 0 && powerSum[r] > 0 {
			avgP := powerSum[r] / float64(powerCount[r])
			lf := math.Log(float64(r))
			lp := math.Log(avgP)
			sumLF += lf
			sumLP += lp
			sumLF2 += lf * lf
			sumLFLP += lf * lp
			nPts++
		}
	}
	if nPts <= 2 {
		return 2.0
	}
	denom := float64(nPts)*sumLF2 - sumLF*sumLF
	if math.Abs(denom) < 1e-10 {
		return 2.0
	}
	slope := (float64(nPts)*sumLFLP - sumLF*sumLP) / denom
	return clampF(-slope, 0.5, 10.0)
}

func channelCorrelationOf(img *image.RGBA) float64 {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	var rVals, gVals, bVals []float64
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			rVals = append(rVals, float64(r>>8))
			gVals = append(gVals, float64(g>>8))
			bVals = append(bVals, float64(b>>8))
		}
	}
	return clampF((pearsonCorr(rVals, gVals)+pearsonCorr(rVals, bVals))/2, 0, 1)
}

// reverseScore measures how well a generated image matches the target's
// *statistical character*: spectral slope + channel correlation, mixed with
// raw MSE. Random-phase spectra can never match pixel-for-pixel, so pure MSE
// is a nearly hopeless objective — this finds genomes whose renders LOOK like
// the target even with a different seed. Returns (structural score, MSE);
// MSE is returned separately so the UI similarity % stays comparable.
func reverseScore(target, cand *image.RGBA) (score, mse float64) {
	mse = compareImages(target, cand)
	slopeDiff := math.Abs(estimateExponent(target) - estimateExponent(cand))
	corrDiff := math.Abs(channelCorrelationOf(target) - channelCorrelationOf(cand))
	return mse/(255.0*255.0) + 0.3*slopeDiff + 0.3*corrDiff, mse
}

func perturbGenome(g Genome, rng *rand.Rand) Genome {
	candidate := g
	step := rng.Float64()*2 - 1
	switch rng.Intn(17) {
	case 0:
		candidate.Exponent = clampF(candidate.Exponent+step*0.3, 0.5, 10.0)
	case 1:
		candidate.BandLimit = clampF(candidate.BandLimit+step*0.1, 0.01, 1.0)
	case 2:
		candidate.AxisStretch = clampF(candidate.AxisStretch+step*0.15, 0.25, 4.0)
	case 3:
		candidate.Gamma = clampF(candidate.Gamma+step*0.15, 0.3, 3.0)
	case 4:
		candidate.Colorfulness = clampF(candidate.Colorfulness+step*0.1, 0, 1)
	case 5:
		candidate.PalA = jitter3(candidate.PalA, 0.05, rng)
		candidate.PalB = jitter3(candidate.PalB, 0.05, rng)
	case 6:
		candidate.PalC = jitter3(candidate.PalC, 0.05, rng)
		candidate.PalD = jitter3(candidate.PalD, 0.05, rng)
	case 7:
		candidate.ExponentHi = clampF(candidate.ExponentHi+step*0.3, 0.5, 10.0)
		candidate.BreakFreq = clampF(candidate.BreakFreq+step*0.08, 0.0, 0.9)
	case 7 + 1:
		candidate.ReliefAngle += step * 0.4
		candidate.ReliefStrength = clampF(candidate.ReliefStrength+step*0.1, 0, 2)
	case 9:
		candidate.Transform = rng.Intn(4)
		if candidate.Transform == 2 || candidate.Transform == 3 {
			candidate.TerraceLevels = 3.0 + rng.Float64()*12.0
		}
	case 10:
		candidate.SpikeCount = 1 + rng.Intn(4)
		candidate.SpikeAmp = clampF(candidate.SpikeAmp+step*3.0, 0.5, 20.0)
	case 11:
		candidate.ChromaStrength = clampF(candidate.ChromaStrength+step*0.1, 0, 0.8)
	case 12:
		candidate.SpecRot += step * 0.8
		candidate.ConeAngle += step * 0.8
		if rng.Float64() < 0.3 {
			if candidate.ConeWidth > 0.999 {
				candidate.ConeWidth = 0.25 + rng.Float64()*0.5
			} else {
				candidate.ConeWidth = 1.0
			}
		}
	case 13:
		candidate.DomainWarp = clampF(candidate.DomainWarp+step*0.08, 0, 0.5)
	case 14:
		candidate.NormMode = rng.Intn(3)
	case 15:
		candidate.SymmetryFold = rng.Intn(9)
		if rng.Float64() < 0.4 {
			candidate.SymmetryMirror = !candidate.SymmetryMirror
		}
	case 16:
		candidate.PaletteMode = rng.Intn(2)
		if candidate.PaletteMode == 1 && candidate.AnchorCount < 2 {
			candidate.AnchorCount = 2 + rng.Intn(4)
			for k := 0; k < candidate.AnchorCount; k++ {
				for i := 0; i < 3; i++ {
					candidate.AnchorColors[k][i] = rng.Float64()
				}
			}
		}
	}
	if rng.Float64() < 0.1 {
		candidate.Seed = rng.Int63()
	}
	return candidate
}

// ============================================================================
// REVERSE ENGINEERING — METHOD 1: BRUTE FORCE
// ============================================================================

// ============================================================================
// REVERSE ENGINEERING v2 — DETERMINISTIC ANALYSIS & PHASE MATCH
// ============================================================================

type palPt struct{ t, v float64 }

// estimateAxisStretch compares spectral spread along fu vs fv.
// The synthesis stretches fv by s, so the ratio of the second moments
// recovers s directly.
func estimateAxisStretch(img *image.RGBA) float64 {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	padW, padH := nextPow2(w), nextPow2(h)
	data := make([][]complex128, padH)
	for y := 0; y < padH; y++ {
		data[y] = make([]complex128, padW)
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			gray := float64(r+g+b) / 3.0 / 65535.0
			data[y][x] = complex(gray*2-1, 0)
		}
	}
	fft2d(data, false)
	var sw, swx2, swy2 float64
	for y := 0; y < padH; y++ {
		for x := 0; x < padW; x++ {
			fu := freqCoord(x, padW)
			fv := freqCoord(y, padH)
			p := real(data[y][x])*real(data[y][x]) + imag(data[y][x])*imag(data[y][x])
			sw += p
			swx2 += p * fu * fu
			swy2 += p * fv * fv
		}
	}
	if sw < 1e-12 {
		return 1.0
	}
	return clampF(math.Sqrt(swy2/sw)/math.Sqrt(swx2/sw), 0.25, 4.0)
}

// collectPalettePoints returns (normalized luminance, channel value) samples,
// sorted by t so the fit sees the palette as a coherent curve.
func collectPalettePoints(img *image.RGBA) (pts [3][]palPt) {
	b := img.Bounds()
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			r, g, b2, _ := img.At(x, y).RGBA()
			fr, fg, fb := float64(r>>8), float64(g>>8), float64(b2>>8)
			t := (fr + fg + fb) / 3.0 / 255.0 // same gray the renderer uses
			pts[0] = append(pts[0], palPt{t, fr / 255})
			pts[1] = append(pts[1], palPt{t, fg / 255})
			pts[2] = append(pts[2], palPt{t, fb / 255})
		}
	}
	for ch := 0; ch < 3; ch++ {
		sort.Slice(pts[ch], func(i, j int) bool { return pts[ch][i].t < pts[ch][j].t })
	}
	return
}

// fitChannel fits y ~= a + b*cos(2pi*(c*x + d)) by grid search over (c, d)
// with a closed-form linear solve for (a, b) at each grid point.
func fitChannel(pts []palPt, gamma float64) (a, b, c, d, bestErr float64) {
	n := len(pts)
	xs := make([]float64, n)
	for i, p := range pts {
		xs[i] = math.Pow(clampF(p.t, 0, 1), gamma)
	}
	eval := func(cc, dd float64) (aa, bb, e float64) {
		var s0, s1, s11, t0, t1 float64
		for i := range xs {
			c1 := math.Cos(2 * math.Pi * (cc*xs[i] + dd))
			s0++
			s1 += c1
			s11 += c1 * c1
			t0 += pts[i].v
			t1 += pts[i].v * c1
		}
		den := s0*s11 - s1*s1
		if math.Abs(den) < 1e-9 {
			return 0, 0, math.MaxFloat64
		}
		aa = (t0*s11 - t1*s1) / den
		bb = (s0*t1 - s1*t0) / den
		for i := range xs {
			pred := aa + bb*math.Cos(2*math.Pi*(cc*xs[i]+dd))
			dv := pts[i].v - pred
			e += dv * dv
		}
		return aa, bb, e
	}
	bestErr = math.MaxFloat64
	for ci := 0; ci <= 20; ci++ {
		cc := float64(ci) * 0.1
		for dj := 0; dj < 16; dj++ {
			dd := float64(dj) / 16.0
			aa, bb, e := eval(cc, dd)
			if e < bestErr {
				bestErr, a, b, c, d = e, aa, bb, cc, dd
			}
		}
	}
	c0, d0 := c, d
	for ci := -5; ci <= 5; ci++ {
		cc := c0 + float64(ci)*0.01
		for dj := -8; dj <= 8; dj++ {
			dd := math.Mod(d0+float64(dj)/128.0+2.0, 1.0)
			aa, bb, e := eval(cc, dd)
			if e < bestErr {
				bestErr, a, b, c, d = e, aa, bb, cc, dd
			}
		}
	}
	return
}

// fitPaletteGamma picks gamma and the full cosine palette jointly.
func fitPaletteGamma(pts [3][]palPt) (a, b, c, d [3]float64, gamma float64) {
	bestTotal := math.MaxFloat64
	store := func(pal [3][4]float64, g float64) {
		gamma = g
		for ch := 0; ch < 3; ch++ {
			a[ch], b[ch], c[ch], d[ch] = pal[ch][0], pal[ch][1], pal[ch][2], pal[ch][3]
		}
	}
	tryGamma := func(g float64) ([3][4]float64, float64) {
		var pal [3][4]float64
		total := 0.0
		for ch := 0; ch < 3; ch++ {
			pa, pb, pc, pd, e := fitChannel(pts[ch], g)
			pal[ch] = [4]float64{pa, pb, pc, pd}
			total += e
		}
		return pal, total
	}
	for gam := 0.3; gam <= 2.51; gam += 0.1 {
		pal, tot := tryGamma(gam)
		if tot < bestTotal {
			bestTotal = tot
			store(pal, gam)
		}
	}
	base := gamma
	for _, gam := range []float64{base - 0.08, base - 0.04, base + 0.04, base + 0.08} {
		if gam < 0.3 || gam > 2.5 {
			continue
		}
		pal, tot := tryGamma(gam)
		if tot < bestTotal {
			bestTotal = tot
			store(pal, gam)
		}
	}
	return
}

// analyzeGenome builds a genome from direct measurement of the target.
func analyzeGenome(target *image.RGBA) Genome {
	pts := collectPalettePoints(target)
	palA, palB, palC, palD, gamma := fitPaletteGamma(pts)
	return Genome{
		Seed:          0,
		Exponent:      estimateExponent(target),
		BandLimit:     0.6,
		AxisStretch:   estimateAxisStretch(target),
		Gamma:         gamma,
		Colorfulness:  channelCorrelationOf(target),
		MutationRate:  0,
		MutationPower: 0,
		PalA:          palA, PalB: palB, PalC: palC, PalD: palD,
	}
}

// fitRefSize caps a resolution so its longest side is maxSide,
// preserving aspect ratio.
func fitRefSize(w, h, maxSide int) (int, int) {
	long := w
	if h > long {
		long = h
	}
	if long <= maxSide {
		return w, h
	}
	s := float64(maxSide) / float64(long)
	return int(float64(w)*s + 0.5), int(float64(h)*s + 0.5)
}

// extractLumaRef encodes a downscaled copy of the target as a high-quality
// JPEG. JPEG keeps the reference embedded in the genome compact even at the
// large sizes needed for artifact-free full-resolution exports.
func extractLumaRef(target *image.RGBA, w, h int) (b64 string, lw, lh int) {
	src := downsampleImage(target, w, h)
	var buf bytes.Buffer
	jpeg.Encode(&buf, src, &jpeg.Options{Quality: 88})
	return base64.StdEncoding.EncodeToString(buf.Bytes()), w, h
}

// synthMatchChannel re-textures the stored luma map: keeps the target's FFT
// phase (spatial layout) and replaces amplitudes with the genome's spectral
// shaping. Output buffer is padded to power-of-two so generateSpectralImage's
// downstream indexing (y*padW + x) works unchanged for both paths.
func synthMatchChannel(width, height int, cfg Genome) []float64 {
	raw, err := base64.StdEncoding.DecodeString(cfg.LumaRef)
	if err != nil {
		return nil
	}
	lumaImg, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil
	}
	lw := cfg.LumaW
	lh := cfg.LumaH
	if lw != lumaImg.Bounds().Dx() || lh != lumaImg.Bounds().Dy() {
		lw = lumaImg.Bounds().Dx()
		lh = lumaImg.Bounds().Dy()
	}
	if lw < 2 || lh < 2 {
		return nil
	}

	// Preview/evolution renders happen at small canvas sizes. Downscale the
	// reference for those so the FFT work (and thus evolution speed) stays
	// cheap. Full-size exports keep the reference at native resolution and
	// therefore render with full detail instead of being upscaled.
	if width < lw/2 {
		scale := float64(width*2) / float64(lw)
		nh := int(float64(lh)*scale + 0.5)
		if nh < 2 {
			nh = 2
		}
		lumaImg = downsampleImage(lumaImg, width*2, nh)
		lw, lh = width*2, nh
	}

	padSrcW := nextPow2(lw)
	padSrcH := nextPow2(lh)

	data := make([][]complex128, padSrcH)
	for y := 0; y < padSrcH; y++ {
		data[y] = make([]complex128, padSrcW)
	}

	zoom := cfg.Zoom
	if zoom <= 0.01 {
		zoom = 1.0
	}
	cosR, sinR := math.Cos(-cfg.Rot), math.Sin(-cfg.Rot)
	s := math.Min(float64(lw), float64(lh))
	fcx := float64(lw)/2.0 + cfg.CenterX*float64(lw)
	fcy := float64(lh)/2.0 + cfg.CenterY*float64(lh)

	// Seed-driven liquid warp: one shared smooth field, deterministic from
	// Seed, amplitude = Warp (fraction of the frame). Identity at Warp = 0.
	warp := clampF(cfg.Warp, 0, 0.5)
	var hfx, hfy, hph [3]float64
	if warp > 0 {
		wr := rand.New(rand.NewSource(cfg.Seed + 7777))
		for h := 0; h < 3; h++ {
			hfx[h] = 1.5 + wr.Float64()*2.5
			hfy[h] = 1.5 + wr.Float64()*2.5
			hph[h] = wr.Float64() * 2 * math.Pi
		}
	}

	for y := 0; y < lh; y++ {
		ny := (float64(y) - float64(lh)/2.0) / s
		for x := 0; x < lw; x++ {
			nx := (float64(x) - float64(lw)/2.0) / s

			rx := (nx*cosR - ny*sinR) / zoom
			ry := (nx*sinR + ny*cosR) / zoom
			if cfg.FlipX > 0.5 {
				rx = -rx
			}
			if cfg.FlipY > 0.5 {
				ry = -ry
			}
			if warp > 0 {
				wd := 0.0
				for h := 0; h < 3; h++ {
					wd += math.Sin(2*math.Pi*(hfx[h]*rx+hfy[h]*ry) + hph[h])
				}
				off := warp * wd / 3.0
				rx += off
				ry += off * 0.8
			}

			sxi := int(rx*s + fcx + 0.5)
			syi := int(ry*s + fcy + 0.5)
			sxi = ((sxi % lw) + lw) % lw
			syi = ((syi % lh) + lh) % lh

			r, g, b, _ := lumaImg.At(sxi, syi).RGBA()
			gray := (float64(r>>8) + float64(g>>8) + float64(b>>8)) / 3.0 / 255.0
			data[y][x] = complex(gray*2-1, 0)
		}
	}
	fft2d(data, false)

	mix := clampF(cfg.PhaseMix, 0, 1)
	var jitRng *rand.Rand
	if cfg.PhaseJitter > 0 {
		jitRng = rand.New(rand.NewSource(cfg.Seed))
	}
	for y := 0; y < padSrcH; y++ {
		for x := 0; x < padSrcW; x++ {
			re, im := real(data[y][x]), imag(data[y][x])
			origMag := math.Hypot(re, im)
			if origMag < 1e-12 {
				continue
			}
			fu := freqCoord(x, padSrcW)
			fv := freqCoord(y, padSrcH)
			fvv := fv * cfg.AxisStretch
			f := math.Sqrt(fu*fu + fvv*fvv)

			targetAmp := origMag
			if f >= 0.5 {
				desired := 1.0 / math.Pow(f, cfg.Exponent/2.0)
				targetAmp = (origMag + desired) / 2.0
			}
			if cfg.BandLimit > 0 {
				cutoff := float64(padSrcW/2) * cfg.BandLimit
				if f > cutoff {
					targetAmp *= math.Exp(-((f - cutoff) * (f - cutoff)) / (2 * cutoff * cutoff))
				}
			}
			ph := math.Atan2(im, re) * mix
			if jitRng != nil {
				ph += (jitRng.Float64()*2 - 1) * cfg.PhaseJitter
			}
			data[y][x] = complex(targetAmp*math.Cos(ph), targetAmp*math.Sin(ph))
		}
	}
	fft2d(data, true)

	padOutW := nextPow2(width)
	padOutH := nextPow2(height)
	out := make([]float64, padOutW*padOutH)
	for y := 0; y < height; y++ {
		fy := float64(y) * float64(lh-1) / float64(height-1)
		y0 := int(fy)
		dy := fy - float64(y0)
		y1 := minInt(y0+1, lh-1)
		for x := 0; x < width; x++ {
			fx := float64(x) * float64(lw-1) / float64(width-1)
			x0 := int(fx)
			dx := fx - float64(x0)
			x1 := minInt(x0+1, lw-1)
			top := real(data[y0][x0])*(1-dx) + real(data[y0][x1])*dx
			bot := real(data[y1][x0])*(1-dx) + real(data[y1][x1])*dx
			out[y*padOutW+x] = top*(1-dy) + dy*bot
		}
	}
	return out
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func reverseEngineerPrecise(target *image.RGBA, refineIter int) (Genome, float64) {
	const cmpW, cmpH = 128, 96
	cmpTarget := downsampleImage(target, cmpW, cmpH)
	tgtExp := estimateExponent(cmpTarget)
	tgtCorr := channelCorrelationOf(cmpTarget)
	scoreWith := func(cand *image.RGBA) (float64, float64) {
		mse := compareImages(cmpTarget, cand)
		sd := math.Abs(estimateExponent(cand) - tgtExp)
		cd := math.Abs(channelCorrelationOf(cand) - tgtCorr)
		return mse/(255.0*255.0) + 0.3*sd + 0.3*cd, mse
	}

	genome := analyzeGenome(cmpTarget)

	// Probe several seeds: with random phases only the texture character can
	// match, so pick the luckiest realization.
	searchRng := rand.New(rand.NewSource(time.Now().UnixNano()))
	bestScore := math.MaxFloat64
	bestMSE := math.MaxFloat64
	for s := 0; s < 8; s++ {
		g := genome
		g.Seed = searchRng.Int63()
		genImg := generateSpectralImage(cmpW, cmpH, rand.New(rand.NewSource(g.Seed)), g)
		sc, mse := scoreWith(genImg)
		if sc < bestScore {
			bestScore, bestMSE, genome = sc, mse, g
		}
	}
	// Short hill climb to polish.
	for iter := 0; iter < refineIter; iter++ {
		cand := perturbGenome(genome, searchRng)
		genImg := generateSpectralImage(cmpW, cmpH, rand.New(rand.NewSource(cand.Seed)), cand)
		sc, mse := scoreWith(genImg)
		if sc < bestScore {
			bestScore, bestMSE, genome = sc, mse, cand
		}
	}
	return genome, bestMSE
}

func reverseEngineerMatch(target image.Image, refineIter int) (Genome, float64) {
	const cmpW, cmpH = 128, 96
	cmpTarget := downsampleImage(target, cmpW, cmpH)

	genome := analyzeGenome(cmpTarget)
	// Store a high-resolution luma reference (capped at 1920 px on the long
	// side) so exports render natively at the requested size instead of
	// being upscaled from a small map.
	refW, refH := fitRefSize(target.Bounds().Dx(), target.Bounds().Dy(), 1920)
	if refW < 2 || refH < 2 {
		refW, refH = cmpW, cmpH
	}
	lumaB64, lw, lh := extractLumaRef(downsampleImage(target, refW, refH), refW, refH)
	genome.LumaRef, genome.LumaW, genome.LumaH = lumaB64, lw, lh
	genome.PhaseMix = 1.0
	genome.Structure = 1.0

	tgtExp := estimateExponent(cmpTarget)
	tgtCorr := channelCorrelationOf(cmpTarget)
	scoreOf := func(cand Genome) float64 {
		img := generateSpectralImage(cmpW, cmpH, rand.New(rand.NewSource(cand.Seed)), cand)
		mse := compareImages(cmpTarget, img)
		sd := math.Abs(estimateExponent(img) - tgtExp)
		cd := math.Abs(channelCorrelationOf(img) - tgtCorr)
		return mse/(255.0*255.0) + 0.3*sd + 0.3*cd
	}

	bestScore := scoreOf(genome)
	searchRng := rand.New(rand.NewSource(time.Now().UnixNano()))
	for iter := 0; iter < refineIter; iter++ {
		cand := perturbGenome(genome, searchRng)
		// Never let the search drop the phase reference.
		cand.LumaRef, cand.LumaW, cand.LumaH = genome.LumaRef, genome.LumaW, genome.LumaH
		cand.PhaseMix = genome.PhaseMix
		if sc := scoreOf(cand); sc < bestScore {
			bestScore, genome = sc, cand
		}
	}
	return genome, bestScore
}

func reverseEngineerBruteForce(target *image.RGBA, iterations int) (Genome, float64) {
	const cmpW, cmpH = 64, 48
	cmpTarget := downsampleImage(target, cmpW, cmpH)
	// Cached target statistics (were recomputed on every iteration).
	tgtExp := estimateExponent(cmpTarget)
	tgtCorr := channelCorrelationOf(cmpTarget)
	scoreWith := func(cand *image.RGBA) (float64, float64) {
		mse := compareImages(cmpTarget, cand)
		sd := math.Abs(estimateExponent(cand) - tgtExp)
		cd := math.Abs(channelCorrelationOf(cand) - tgtCorr)
		return mse/(255.0*255.0) + 0.3*sd + 0.3*cd, mse
	}
	bestGenome := Genome{}
	bestScore := math.MaxFloat64
	bestMSE := math.MaxFloat64
	for i := 0; i < iterations; i++ {
		seedRng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(i)*999999))
		genome := randomGenome(seedRng)
		imgRng := rand.New(rand.NewSource(genome.Seed))
		genImg := generateSpectralImage(cmpW, cmpH, imgRng, genome)
		score, mse := scoreWith(genImg)
		if score < bestScore {
			bestScore = score
			bestGenome = genome
			bestMSE = mse
		}
	}
	return bestGenome, bestMSE
}

// ============================================================================
// REVERSE ENGINEERING — METHOD 2: HILL CLIMBING
// ============================================================================

func reverseEngineerHillClimb(target *image.RGBA, iterations int) (Genome, float64) {
	const cmpW, cmpH = 64, 48
	cmpTarget := downsampleImage(target, cmpW, cmpH)
	tgtExp := estimateExponent(cmpTarget)
	tgtCorr := channelCorrelationOf(cmpTarget)
	scoreWith := func(cand *image.RGBA) (float64, float64) {
		mse := compareImages(cmpTarget, cand)
		sd := math.Abs(estimateExponent(cand) - tgtExp)
		cd := math.Abs(channelCorrelationOf(cand) - tgtCorr)
		return mse/(255.0*255.0) + 0.3*sd + 0.3*cd, mse
	}
	searchRng := rand.New(rand.NewSource(time.Now().UnixNano()))
	genome := randomGenome(searchRng)
	imgRng := rand.New(rand.NewSource(genome.Seed))
	genImg := generateSpectralImage(cmpW, cmpH, imgRng, genome)
	bestScore, bestMSE := scoreWith(genImg)
	noImproveCount := 0
	for iter := 0; iter < iterations; iter++ {
		candidate := perturbGenome(genome, searchRng)
		imgRng := rand.New(rand.NewSource(candidate.Seed))
		genImg := generateSpectralImage(cmpW, cmpH, imgRng, candidate)
		score, mse := scoreWith(genImg)
		if score < bestScore {
			bestScore = score
			bestMSE = mse
			genome = candidate
			noImproveCount = 0
		} else {
			noImproveCount++
			if noImproveCount > 100 {
				genome = randomGenome(searchRng)
				imgRng := rand.New(rand.NewSource(genome.Seed))
				genImg := generateSpectralImage(cmpW, cmpH, imgRng, genome)
				bestScore, bestMSE = scoreWith(genImg)
				noImproveCount = 0
			}
		}
	}
	return genome, bestMSE
}

// ============================================================================
// REVERSE ENGINEERING — METHOD 3: ESTIMATE + FINE-TUNE
// ============================================================================

func reverseEngineerEstimate(target *image.RGBA, fineTuneIter int) (Genome, float64) {
	const cmpW, cmpH = 64, 48
	cmpTarget := downsampleImage(target, cmpW, cmpH)
	tgtExp := estimateExponent(cmpTarget)
	tgtCorr := channelCorrelationOf(cmpTarget)
	scoreWith := func(cand *image.RGBA) (float64, float64) {
		mse := compareImages(cmpTarget, cand)
		sd := math.Abs(estimateExponent(cand) - tgtExp)
		cd := math.Abs(channelCorrelationOf(cand) - tgtCorr)
		return mse/(255.0*255.0) + 0.3*sd + 0.3*cd, mse
	}

	genome := Genome{
		Seed:          time.Now().UnixNano(),
		Exponent:      tgtExp,
		BandLimit:     0.5,
		AxisStretch:   1.0,
		Gamma:         1.0,
		Colorfulness:  tgtCorr,
		MutationRate:  0.0,
		MutationPower: 0.0,
	}

	searchRng := rand.New(rand.NewSource(time.Now().UnixNano()))
	imgRng := rand.New(rand.NewSource(genome.Seed))
	genImg := generateSpectralImage(cmpW, cmpH, imgRng, genome)
	bestScore, bestMSE := scoreWith(genImg)
	noImproveCount := 0
	for iter := 0; iter < fineTuneIter; iter++ {
		candidate := perturbGenome(genome, searchRng)
		imgRng := rand.New(rand.NewSource(candidate.Seed))
		genImg := generateSpectralImage(cmpW, cmpH, imgRng, candidate)
		score, mse := scoreWith(genImg)
		if score < bestScore {
			bestScore = score
			bestMSE = mse
			genome = candidate
			noImproveCount = 0
		} else {
			noImproveCount++
			if noImproveCount > 80 {
				genome = randomGenome(searchRng)
				imgRng := rand.New(rand.NewSource(genome.Seed))
				genImg := generateSpectralImage(cmpW, cmpH, imgRng, genome)
				bestScore, bestMSE = scoreWith(genImg)
				noImproveCount = 0
			}
		}
	}
	return genome, bestMSE
}

func pearsonCorr(a, b []float64) float64 {
	n := float64(len(a))
	if n == 0 {
		return 0
	}
	var sa, sb float64
	for i := range a {
		sa += a[i]
		sb += b[i]
	}
	ma := sa / n
	mb := sb / n
	var cov, va, vb float64
	for i := range a {
		da := a[i] - ma
		db := b[i] - mb
		cov += da * db
		va += da * da
		vb += db * db
	}
	if va < 1e-10 || vb < 1e-10 {
		return 0
	}
	return cov / math.Sqrt(va*vb)
}

// ============================================================================
// EASING FUNCTIONS
// ============================================================================

// easeLinear - No easing, constant speed
func easeLinear(t float64) float64 { return t }

func easeInQuad(t float64) float64 { return t * t }

func easeOutQuad(t float64) float64 { return t * (2.0 - t) }

func easeInOutQuad(t float64) float64 {
	if t < 0.5 {
		return 2.0 * t * t
	}
	return 1.0 - math.Pow(-2.0*t+2.0, 2.0)/2.0
}

func easeInCubic(t float64) float64 { return t * t * t }

func easeOutCubic(t float64) float64 { return math.Pow(t-1.0, 3.0) + 1.0 }

func easeInOutCubic(t float64) float64 {
	c2 := math.Pow(2.0, 2.0) - 1.0
	if t < 0.5 {
		return math.Pow(2.0*t, 3.0) / 2.0
	}
	return (math.Pow(2.0*t-2.0, 3.0) + c2 + 1.0) / 2.0
}

func easeInQuart(t float64) float64 { return t * t * t * t }

func easeInOutQuart(t float64) float64 {
	if t < 0.5 {
		return 8.0 * t * t * t * t
	}
	return 1.0 - math.Pow(-2.0*t+2.0, 4.0)/2.0
}

func easeOutQuart(t float64) float64 {
	return 1.0 - math.Pow(-2.0*t+2.0, 4.0)/2.0
}

func easeInQuint(t float64) float64 { return t * t * t * t * t }

func easeOutQuint(t float64) float64 { return math.Pow(t-1.0, 5.0) + 1.0 }

func easeInOutQuint(t float64) float64 {
	c1 := 1.0
	c2 := c1 - 1.0
	if t < 0.5 {
		return math.Pow(2.0*t, 5.0) / 2.0
	}
	return (math.Pow(2.0*t-2.0, 5.0) + c1 + c2) / 2.0
}

func easeSin(t float64) float64 { return (1.0 - math.Cos(t*math.Pi)) / 2.0 }

func easeInSin(t float64) float64 { return 1.0 - math.Cos(t*math.Pi/2.0) }

func easeOutSin(t float64) float64 { return math.Sin(t * math.Pi / 2.0) }

func easeInOutSin(t float64) float64 { return -(math.Cos(math.Pi*t) - 1.0) / 2.0 }

func easeExpoIn(t float64) float64 {
	if t == 0 {
		return 0
	}
	return math.Pow(2.0, 10.0*t-10.0)
}

func easeExpoOut(t float64) float64 {
	if t == 1 {
		return 1
	}
	return 1.0 - math.Pow(2.0, -10.0*t)
}

func easeExpoInOut(t float64) float64 {
	if t == 0 {
		return 0
	}
	if t == 1 {
		return 1
	}
	if t < 0.5 {
		return math.Pow(2.0, 20.0*t-10.0) / 2.0
	}
	return (2.0 - math.Pow(2.0, -20.0*t+10.0)) / 2.0
}

func easeCircleIn(t float64) float64 { return 1.0 - math.Sqrt(1.0-t*t) }

func easeCircleOut(t float64) float64 { return math.Sqrt(1.0 - (t-1.0)*(t-1.0)) }

func easeCircleInOut(t float64) float64 {
	if t < 0.5 {
		return (1.0 - math.Sqrt(1.0-4.0*(t*0.5)*(t*0.5))) / 2.0
	}
	return (math.Sqrt(1.0-(-2.0*t+2.0)*(-2.0*t+2.0)) + 1.0) / 2.0
}

func easeBackIn(t float64) float64 {
	c1 := 1.70158
	c3 := c1 + 1.0
	return c3*t*t*t - c1*t*t
}

func easeBackOut(t float64) float64 {
	c1 := 1.70158
	c3 := c1 + 1.0
	return 1.0 + c3*math.Pow(t-1.0, 3.0) + c1*math.Pow(t-1.0, 2.0)
}

func easeBackInOut(t float64) float64 {
	c1 := 1.70158
	c2 := c1 * 1.525
	if t < 0.5 {
		return (t * 2.0) * (t * 2.0) * ((c2+1.0)*t*2.0 - c2) / 2.0
	}
	return ((t*2.0-2.0)*(t*2.0-2.0)*((c2+1.0)*(t*2.0-2.0)+c2) + 2.0) / 2.0
}

func easeElasticIn(t float64) float64 {
	if t == 0 || t == 1 {
		return t
	}
	c4 := (2.0 * math.Pi) / 3.0
	return -math.Pow(2.0, 10.0*t-10.0) * math.Sin((t*10.0-10.75)*c4)
}

func easeElasticOut(t float64) float64 {
	if t == 0 || t == 1 {
		return t
	}
	c4 := (2.0 * math.Pi) / 3.0
	return math.Pow(2.0, -10.0*t)*math.Sin((t*10.0-0.75)*c4) + 1.0
}

func easeElasticInOut(t float64) float64 {
	if t == 0 {
		return 0
	}
	if t == 1 {
		return 1
	}
	c4 := (2.0 * math.Pi) / 3.0
	if t < 0.5 {
		return -(math.Pow(2.0, 20.0*t-10.0) * math.Sin((t*20.0-11.125)*c4)) / 2.0
	}
	return (math.Pow(2.0, -20.0*t+10.0)*math.Sin((t*20.0-11.125)*c4))/2.0 + 1.0
}

func easeBounceOut(t float64) float64 {
	n1 := 7.5625
	d1 := 2.75
	if t < 1.0/d1 {
		return n1 * t * t
	} else if t < 2.0/d1 {
		t -= 1.5 / d1
		return n1*t*t + 0.75
	} else if t < 2.5/d1 {
		t -= 2.25 / d1
		return n1*t*t + 0.9375
	}
	t -= 2.625 / d1
	return n1*t*t + 0.984375
}

func easeBounceIn(t float64) float64 { return 1.0 - easeBounceOut(1.0-t) }

func easeBounceInOut(t float64) float64 {
	if t < 0.5 {
		return (1.0 - easeBounceOut(1.0-2.0*t)) / 2.0
	}
	return (1.0 + easeBounceOut(2.0*t-1.0)) / 2.0
}

func easeGauss(t float64) float64 {
	sigma := 0.25
	mu := 0.5
	val := math.Exp(-math.Pow(t-mu, 2.0) / (2.0 * math.Pow(sigma, 2.0)))
	peak := math.Exp(-math.Pow(mu, 2.0) / (2.0 * math.Pow(sigma, 2.0)))
	return val / peak
}

func getEasingFunc(name string) func(float64) float64 {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "linear":
		return easeLinear
	case "inquad", "easeinquad":
		return easeInQuad
	case "outquad", "easeoutquad":
		return easeOutQuad
	case "inoutquad", "easeinoutquad":
		return easeInOutQuad
	case "incubic", "easeincubic":
		return easeInCubic
	case "outcubic", "easeoutcubic":
		return easeOutCubic
	case "inoutcubic", "easeinoutcubic":
		return easeInOutCubic
	case "inquart", "easeinquart":
		return easeInQuart
	case "outquart", "easeoutquart":
		return easeOutQuart
	case "inoutquart", "easeinoutquart":
		return easeInOutQuart
	case "inquint", "easeinquint":
		return easeInQuint
	case "outquint", "easeoutquint":
		return easeOutQuint
	case "inoutquint", "easeinoutquint":
		return easeInOutQuint
	case "sin", "ease":
		return easeSin
	case "insin", "easeinsin":
		return easeInSin
	case "outsin", "easeoutsin":
		return easeOutSin
	case "inoutsin", "easeinoutsin":
		return easeInOutSin
	case "expo":
		return easeExpoInOut
	case "expoin", "easeexpon":
		return easeExpoIn
	case "expoout", "easeexpoout":
		return easeExpoOut
	case "circle", "circ":
		return easeCircleInOut
	case "circlein", "circin":
		return easeCircleIn
	case "circleout", "circout":
		return easeCircleOut
	case "back":
		return easeBackInOut
	case "backin":
		return easeBackIn
	case "backout":
		return easeBackOut
	case "elastic":
		return easeElasticInOut
	case "elasticin":
		return easeElasticIn
	case "elasticout":
		return easeElasticOut
	case "bounce":
		return easeBounceInOut
	case "bouncein":
		return easeBounceIn
	case "bounceout":
		return easeBounceOut
	case "gauss", "gaussian":
		return easeGauss
	default:
		return easeLinear
	}
}

// ============================================================================
// ANIMATION RENDERER
// ============================================================================

func lerp3(a, b [3]float64, t float64) [3]float64 {
	var out [3]float64
	for i := 0; i < 3; i++ {
		out[i] = a[i] + (b[i]-a[i])*t
	}
	return out
}

// lerpAnchors blends anchor color sets; only A's count is rendered, so
// blending the full array is safe for animation.
func lerpAnchors(a, b [5][3]float64, t float64) [5][3]float64 {
	var out [5][3]float64
	for k := 0; k < 5; k++ {
		out[k] = lerp3(a[k], b[k], t)
	}
	return out
}

func interpolateGenomes(genomeA, genomeB Genome, progress float64) Genome {
	out := Genome{
		Seed:           genomeA.Seed,
		Exponent:       genomeA.Exponent + (genomeB.Exponent-genomeA.Exponent)*progress,
		BandLimit:      genomeA.BandLimit + (genomeB.BandLimit-genomeA.BandLimit)*progress,
		AxisStretch:    genomeA.AxisStretch + (genomeB.AxisStretch-genomeA.AxisStretch)*progress,
		Gamma:          genomeA.Gamma + (genomeB.Gamma-genomeA.Gamma)*progress,
		Colorfulness:   genomeA.Colorfulness + (genomeB.Colorfulness-genomeA.Colorfulness)*progress,
		MutationRate:   genomeA.MutationRate + (genomeB.MutationRate-genomeA.MutationRate)*progress,
		MutationPower:  genomeA.MutationPower + (genomeB.MutationPower-genomeA.MutationPower)*progress,
		PalA:           lerp3(genomeA.PalA, genomeB.PalA, progress),
		PalB:           lerp3(genomeA.PalB, genomeB.PalB, progress),
		PalC:           lerp3(genomeA.PalC, genomeB.PalC, progress),
		PalD:           lerp3(genomeA.PalD, genomeB.PalD, progress),
		Transform:      genomeA.Transform,
		TerraceLevels:  genomeA.TerraceLevels + (genomeB.TerraceLevels-genomeA.TerraceLevels)*progress,
		ReliefAngle:    genomeA.ReliefAngle + (genomeB.ReliefAngle-genomeA.ReliefAngle)*progress,
		ReliefStrength: genomeA.ReliefStrength + (genomeB.ReliefStrength-genomeA.ReliefStrength)*progress,
		ExponentHi:     genomeA.ExponentHi + (genomeB.ExponentHi-genomeA.ExponentHi)*progress,
		BreakFreq:      genomeA.BreakFreq + (genomeB.BreakFreq-genomeA.BreakFreq)*progress,
		SpikeCount:     genomeA.SpikeCount,
		SpikeAmp:       genomeA.SpikeAmp + (genomeB.SpikeAmp-genomeA.SpikeAmp)*progress,
		ChromaStrength: genomeA.ChromaStrength + (genomeB.ChromaStrength-genomeA.ChromaStrength)*progress,
		SpecRot:        genomeA.SpecRot + (genomeB.SpecRot-genomeA.SpecRot)*progress,
		ConeAngle:      genomeA.ConeAngle + (genomeB.ConeAngle-genomeA.ConeAngle)*progress,
		ConeWidth:      genomeA.ConeWidth + (genomeB.ConeWidth-genomeA.ConeWidth)*progress,
		DomainWarp:     genomeA.DomainWarp + (genomeB.DomainWarp-genomeA.DomainWarp)*progress,
		NormMode:       genomeA.NormMode,
		SymmetryFold:   genomeA.SymmetryFold,
		SymmetryMirror: genomeA.SymmetryMirror,
		PaletteMode:    genomeA.PaletteMode,
		AnchorCount:    genomeA.AnchorCount,
		AnchorColors:   lerpAnchors(genomeA.AnchorColors, genomeB.AnchorColors, progress),
	}
	if genomeA.LumaRef != "" && genomeA.LumaRef == genomeB.LumaRef {
		out.LumaRef = genomeA.LumaRef
		out.LumaW = genomeA.LumaW
		out.LumaH = genomeA.LumaH
		out.PhaseMix = genomeA.PhaseMix + (genomeB.PhaseMix-genomeA.PhaseMix)*progress
		out.PhaseJitter = genomeA.PhaseJitter + (genomeB.PhaseJitter-genomeA.PhaseJitter)*progress
		out.Zoom = genomeA.Zoom + (genomeB.Zoom-genomeA.Zoom)*progress
		out.Rot = genomeA.Rot + (genomeB.Rot-genomeA.Rot)*progress
		out.FlipX = genomeA.FlipX
		out.FlipY = genomeA.FlipY
		out.CenterX = genomeA.CenterX + (genomeB.CenterX-genomeA.CenterX)*progress
		out.CenterY = genomeA.CenterY + (genomeB.CenterY-genomeA.CenterY)*progress
		out.Structure = genomeA.Structure + (genomeB.Structure-genomeA.Structure)*progress
		out.Warp = genomeA.Warp + (genomeB.Warp-genomeA.Warp)*progress
	}
	return out
}

// blendImages performs a linear pixel crossfade: result = A*(1-t) + B*t
func blendImages(a, b *image.RGBA, t float64) *image.RGBA {
	bounds := a.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	out := image.NewRGBA(image.Rect(0, 0, w, h))

	tb := float32(t)
	ta := float32(1.0) - tb

	ap := a.Pix
	bp := b.Pix
	op := out.Pix

	for i := 0; i < len(ap); i += 4 {
		op[i] = uint8(float32(ap[i])*ta + float32(bp[i])*tb)
		op[i+1] = uint8(float32(ap[i+1])*ta + float32(bp[i+1])*tb)
		op[i+2] = uint8(float32(ap[i+2])*ta + float32(bp[i+2])*tb)
		op[i+3] = 255
	}
	return out
}

func handleRenderAnimation(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		SourceACell int    `json:"source_a_cell"`
		SourceBCell int    `json:"source_b_cell"`
		Frames      int    `json:"frames"`
		FPS         int    `json:"fps"`
		Width       int    `json:"width"`
		Height      int    `json:"height"`
		Easing      string `json:"easing"`
		Mode        string `json:"mode"` // "crossfade" or "params"
		Dir         string `json:"dir"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request: "+err.Error(), http.StatusBadRequest)
		return
	}

	if req.SourceACell < 0 || req.SourceACell >= totalCells ||
		req.SourceBCell < 0 || req.SourceBCell >= totalCells {
		http.Error(w, "Invalid cell index", http.StatusBadRequest)
		return
	}
	if req.Frames < 1 || req.Frames > 10000 {
		http.Error(w, "Frames must be 1-10000", http.StatusBadRequest)
		return
	}
	if req.FPS < 1 || req.FPS > 60 {
		http.Error(w, "FPS must be 1-60", http.StatusBadRequest)
		return
	}
	if req.Width <= 0 || req.Height <= 0 || req.Width > 8192 || req.Height > 8192 {
		http.Error(w, "Invalid resolution (1-8192)", http.StatusBadRequest)
		return
	}

	easing := req.Easing
	if easing == "" {
		easing = "sine"
	}
	mode := req.Mode
	if mode == "" {
		mode = "crossfade"
	}

	saveDir := req.Dir
	if saveDir == "" {
		saveDir = state.genomeDir
	}
	info, err := os.Stat(saveDir)
	if err != nil || !info.IsDir() {
		http.Error(w, "Directory does not exist", http.StatusBadRequest)
		return
	}

	state.mu.RLock()
	genomeA := state.cells[req.SourceACell].Genome
	genomeB := state.cells[req.SourceBCell].Genome
	state.mu.RUnlock()

	timestamp := time.Now().Format("20060102_150405")
	outputDir := filepath.Join(saveDir, fmt.Sprintf("anim_%04d_to_%04d_%s", req.SourceACell, req.SourceBCell, timestamp))
	os.MkdirAll(outputDir, 0755)

	easeFunc := getEasingFunc(easing)

	processedFiles := make([]string, req.Frames)

	if mode == "crossfade" {
		// CROSSFADE MODE: render both endpoints ONCE, blend per frame.
		// Smooth, shake-free, and only 1 FFT render per endpoint.
		genomeAClean := genomeA
		genomeAClean.MutationRate = 0
		genomeAClean.MutationPower = 0
		genomeBClean := genomeB
		genomeBClean.MutationRate = 0
		genomeBClean.MutationPower = 0

		rngA := rand.New(rand.NewSource(genomeAClean.Seed))
		imgA := generateSpectralImage(req.Width, req.Height, rngA, genomeAClean)

		rngB := rand.New(rand.NewSource(genomeBClean.Seed))
		imgB := generateSpectralImage(req.Width, req.Height, rngB, genomeBClean)

		for frame := 0; frame < req.Frames; frame++ {
			t := 0.0
			if req.Frames > 1 {
				t = float64(frame) / float64(req.Frames-1)
			}
			alpha := easeFunc(t)

			blended := blendImages(imgA, imgB, alpha)

			filename := fmt.Sprintf("frame_%05d.png", frame)
			fpath := filepath.Join(outputDir, filename)
			f, err := os.Create(fpath)
			if err != nil {
				http.Error(w, "Failed to create frame: "+err.Error(), http.StatusInternalServerError)
				return
			}
			png.Encode(f, blended)
			f.Close()
			processedFiles[frame] = filename
		}
	} else {
		// PARAMS MODE: interpolate genome parameters with frozen seed
		// and mutations disabled (prevents flicker/shaking).
		for frame := 0; frame < req.Frames; frame++ {
			t := 0.0
			if req.Frames > 1 {
				t = float64(frame) / float64(req.Frames-1)
			}
			eased := easeFunc(t)

			interp := interpolateGenomes(genomeA, genomeB, eased)
			interp.MutationRate = 0
			interp.MutationPower = 0

			imgRng := rand.New(rand.NewSource(interp.Seed))
			img := generateSpectralImage(req.Width, req.Height, imgRng, interp)

			filename := fmt.Sprintf("frame_%05d.png", frame)
			fpath := filepath.Join(outputDir, filename)
			f, err := os.Create(fpath)
			if err != nil {
				http.Error(w, "Failed to create frame: "+err.Error(), http.StatusInternalServerError)
				return
			}
			png.Encode(f, img)
			f.Close()
			processedFiles[frame] = filename
		}
	}

	animJSON := map[string]interface{}{
		"timestamp":    time.Now().Format("2006-01-02 15:04:05"),
		"frames":       req.Frames,
		"fps":          req.FPS,
		"duration_sec": float64(req.Frames) / float64(req.FPS),
		"width":        req.Width,
		"height":       req.Height,
		"easing":       easing,
		"mode":         mode,
		"genomes": map[string]interface{}{
			"a": map[string]interface{}{"cell": req.SourceACell, "genome": genomeA},
			"b": map[string]interface{}{"cell": req.SourceBCell, "genome": genomeB},
		},
		"frame_files": processedFiles,
	}

	jsonPath := filepath.Join(outputDir, "animation.json")
	jsonData, _ := json.MarshalIndent(animJSON, "", "  ")
	os.WriteFile(jsonPath, jsonData, 0644)

	fmt.Printf("[Animation] Rendered %d frames (%s/%s) to %s\n", req.Frames, mode, easing, outputDir)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":       "rendered",
		"total_frames": req.Frames,
		"output_dir":   outputDir,
		"json_path":    jsonPath,
	})
}

// ============================================================================
// GLOBAL STATE
// ============================================================================

type AppState struct {
	mu        sync.RWMutex
	cells     []*Cell
	genomeDir string
}

type Cell struct {
	Locked  bool
	Image   string
	Genome  Genome
	History []Genome // undo stack, newest last, capped at 50
}

var state *AppState

func NewAppState(genomeDir string) *AppState {
	cs := make([]*Cell, totalCells)
	for i := range cs {
		cs[i] = &Cell{}
	}
	return &AppState{cells: cs, genomeDir: genomeDir}
}

// pushHistoryLocked records the cell's current genome on its undo stack.
// Callers must already hold state.mu.
func (s *AppState) pushHistoryLocked(idx int) {
	h := s.cells[idx].History
	if len(h) >= 50 {
		h = h[len(h)-49:]
	}
	s.cells[idx].History = append(h, s.cells[idx].Genome)
}

// ============================================================================
// HTTP HANDLERS
// ============================================================================

func handleStatic(w http.ResponseWriter, r *http.Request) {
	filePath := "." + r.URL.Path
	if r.URL.Path == "/" || r.URL.Path == "/index.html" {
		filePath = "./static/index.html"
	}
	http.ServeFile(w, r, filePath)
}

func handleGrid(w http.ResponseWriter, r *http.Request) {
	state.mu.RLock()
	defer state.mu.RUnlock()
	cells := make([]map[string]interface{}, totalCells)
	for i, cell := range state.cells {
		// Ship the genome WITHOUT its embedded luma reference: match-mode
		// references are large JPEGs and would bloat every grid fetch. The
		// full genome (reference included) is served by /api/get-genome.
		pub := cell.Genome
		pub.LumaRef = ""
		cells[i] = map[string]interface{}{
			"index":  i,
			"locked": cell.Locked,
			"image":  cell.Image,
			"genome": pub,
		}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"cells": cells})
}

func handleEvolve(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ClickedIndex int `json:"clicked_index"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	state.mu.Lock()
	defer state.mu.Unlock()
	parentGenome := state.cells[req.ClickedIndex].Genome

	// Locked cells act as co-parents: their genes flow into the children.
	donors := make([]Genome, 0, totalCells)
	for i := 0; i < totalCells; i++ {
		if i != req.ClickedIndex && state.cells[i].Locked {
			donors = append(donors, state.cells[i].Genome)
		}
	}

	allLocked := true
	for i := 0; i < totalCells; i++ {
		if i != req.ClickedIndex && !state.cells[i].Locked {
			allLocked = false
			break
		}
	}
	if allLocked {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"error": "All others locked"})
		return
	}
	var wg sync.WaitGroup
	for i := 0; i < totalCells; i++ {
		if i == req.ClickedIndex || state.cells[i].Locked {
			continue
		}
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			mutRng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(idx)*1000000))
			childGenome := breedGenome(parentGenome, donors, mutRng)
			applyFeatures(&childGenome)
			img := renderClean(childGenome, imgW, imgH)
			state.pushHistoryLocked(idx)
			state.cells[idx].Genome = childGenome
			state.cells[idx].Image = imgToBase64(img)
		}(i)
	}
	wg.Wait()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "evolved"})
}

func handleGenerateAll(w http.ResponseWriter, r *http.Request) {
	state.mu.Lock()
	defer state.mu.Unlock()
	var wg sync.WaitGroup
	for i := 0; i < totalCells; i++ {
		if state.cells[i].Locked {
			continue
		}
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			seedRng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(idx)*1000000))
			genome := randomGenome(seedRng)
			applyFeatures(&genome)
			img := renderClean(genome, imgW, imgH)
			state.pushHistoryLocked(idx)
			state.cells[idx].Genome = genome
			state.cells[idx].Image = imgToBase64(img)
		}(i)
	}
	wg.Wait()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "generated"})
}

func handleToggleLock(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Index int `json:"index"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	state.mu.Lock()
	state.cells[req.Index].Locked = !state.cells[req.Index].Locked
	locked := state.cells[req.Index].Locked
	state.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"locked": locked})
}

func handleSaveImage(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Index int    `json:"index"`
		Size  string `json:"size"`
		Dir   string `json:"dir"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	state.mu.RLock()
	g := state.cells[req.Index].Genome
	state.mu.RUnlock()
	parts := strings.Split(req.Size, "x")
	if len(parts) != 2 {
		http.Error(w, "Invalid size format", http.StatusBadRequest)
		return
	}
	width, _ := strconv.Atoi(parts[0])
	height, _ := strconv.Atoi(parts[1])
	if width <= 0 || height <= 0 || width > 8192 || height > 8192 {
		http.Error(w, "Invalid dimensions", http.StatusBadRequest)
		return
	}
	saveDir := req.Dir
	if saveDir == "" {
		saveDir = state.genomeDir
	}
	info, err := os.Stat(saveDir)
	if err != nil || !info.IsDir() {
		http.Error(w, "Directory does not exist", http.StatusBadRequest)
		return
	}
	img := renderPreviewFramed(g, width, height)
	timestamp := time.Now().Format("20060102_150405")
	filename := filepath.Join(saveDir, fmt.Sprintf("image_%04d_%dx%d_%s.png", req.Index, img.Bounds().Dx(), img.Bounds().Dy(), timestamp))
	file, err := os.Create(filename)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer file.Close()
	png.Encode(file, img)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "saved", "path": filename})
}

// handleRenderImage renders a cell at the requested size and streams the PNG
// back to the browser, so the client can save it with a native save dialog.
func handleRenderImage(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Index int    `json:"index"`
		Size  string `json:"size"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Index < 0 || req.Index >= totalCells {
		http.Error(w, "Invalid cell index", http.StatusBadRequest)
		return
	}
	parts := strings.Split(strings.ToLower(strings.TrimSpace(req.Size)), "x")
	if len(parts) != 2 {
		http.Error(w, "Invalid size format (use WxH)", http.StatusBadRequest)
		return
	}
	width, _ := strconv.Atoi(parts[0])
	height, _ := strconv.Atoi(parts[1])
	if width <= 0 || height <= 0 || width > 8192 || height > 8192 {
		http.Error(w, "Invalid dimensions", http.StatusBadRequest)
		return
	}
	state.mu.RLock()
	g := state.cells[req.Index].Genome
	state.mu.RUnlock()

	img := renderPreviewFramed(g, width, height)

	w.Header().Set("Content-Type", "image/png")
	if err := png.Encode(w, img); err != nil {
		http.Error(w, "PNG encode failed: "+err.Error(), http.StatusInternalServerError)
	}
}

func handleSaveParams(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Index int    `json:"index"`
		Dir   string `json:"dir"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	state.mu.RLock()
	g := state.cells[req.Index].Genome
	state.mu.RUnlock()
	saved := SavedGenome{
		Version:   "1.3",
		CellIndex: req.Index,
		Timestamp: time.Now().Format("2006-01-02 15:04:05"),
		Genome:    g,
	}
	jsonData, _ := json.MarshalIndent(saved, "", "  ")
	saveDir := req.Dir
	if saveDir == "" {
		saveDir = state.genomeDir
	}
	timestamp := time.Now().Format("20060102_150405")
	filename := filepath.Join(saveDir, fmt.Sprintf("genome_%04d_%s.json", req.Index, timestamp))
	os.WriteFile(filename, jsonData, 0644)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "saved", "path": filename})
}

func handleLoadParams(w http.ResponseWriter, r *http.Request) {
	// New flow: client picked the file itself and sends its contents.
	if strings.HasPrefix(r.Header.Get("Content-Type"), "application/json") {
		var req struct {
			Index int         `json:"index"`
			Saved SavedGenome `json:"saved"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
			return
		}
		if req.Index < 0 || req.Index >= totalCells {
			http.Error(w, "Invalid cell index", http.StatusBadRequest)
			return
		}
		loadGenomeIntoCell(w, req.Index, req.Saved)
		return
	}

	// Legacy flow: client passed a server-side file path.
	var req struct {
		Index int    `json:"index"`
		Path  string `json:"path"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	if req.Index < 0 || req.Index >= totalCells {
		http.Error(w, "Invalid cell index", http.StatusBadRequest)
		return
	}
	data, err := os.ReadFile(req.Path)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	var saved SavedGenome
	if err := json.Unmarshal(data, &saved); err != nil {
		http.Error(w, "Failed to parse JSON", http.StatusBadRequest)
		return
	}
	loadGenomeIntoCell(w, req.Index, saved)
}

// Shared tail of load-params: validate version, install genome, render preview.
func loadGenomeIntoCell(w http.ResponseWriter, index int, saved SavedGenome) {
	if saved.Version != "1.1" && saved.Version != "1.2" && saved.Version != "1.3" {
		http.Error(w, "Unsupported version (expected 1.1-1.3)", http.StatusBadRequest)
		return
	}
	state.mu.Lock()
	state.cells[index].Genome = saved.Genome
	state.mu.Unlock()
	img := renderClean(saved.Genome, imgW, imgH)
	state.mu.Lock()
	state.cells[index].Image = imgToBase64(img)
	state.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "loaded"})
}

func handleUploadImage(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse multipart form (max 32MB)
	err := r.ParseMultipartForm(32 << 20)
	if err != nil {
		http.Error(w, "Failed to parse upload: "+err.Error(), http.StatusBadRequest)
		return
	}

	cellIndex, err := strconv.Atoi(r.FormValue("cellIndex"))
	if err != nil || cellIndex < 0 || cellIndex >= totalCells {
		http.Error(w, "Invalid cell index", http.StatusBadRequest)
		return
	}

	method := r.FormValue("method")
	if method == "" {
		method = "brute"
	}

	iterationsStr := r.FormValue("iterations")
	iterations := 500
	if iterationsStr != "" {
		if v, err := strconv.Atoi(iterationsStr); err == nil && v > 0 {
			iterations = v
		}
	}

	file, header, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "No image file uploaded", http.StatusBadRequest)
		return
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		http.Error(w, "Failed to decode image: "+err.Error(), http.StatusBadRequest)
		return
	}

	fmt.Printf("[Upload] Processing '%s' (%dx%d) with method '%s' (%d iterations)\n",
		header.Filename, img.Bounds().Dx(), img.Bounds().Dy(), method, iterations)

	downsampled := downsampleImage(img, imgW, imgH)

	var genome Genome
	var score float64

	switch method {
	case "brute":
		genome, score = reverseEngineerBruteForce(downsampled, iterations)
	case "hillclimb":
		genome, score = reverseEngineerHillClimb(downsampled, iterations)
	case "estimate":
		genome, score = reverseEngineerEstimate(downsampled, iterations)
	case "precise":
		genome, score = reverseEngineerPrecise(downsampled, iterations)
	case "match":
		// Pass the original resolution so the match method can embed a
		// high-detail luma reference in the genome.
		genome, score = reverseEngineerMatch(img, iterations)
	default:
		http.Error(w, "Unknown method. Use: brute, hillclimb, or estimate", http.StatusBadRequest)
		return
	}

	displayImg := renderClean(genome, imgW, imgH)

	state.mu.Lock()
	state.pushHistoryLocked(cellIndex)
	state.cells[cellIndex].Genome = genome
	state.cells[cellIndex].Image = imgToBase64(displayImg)
	state.mu.Unlock()

	similarity := math.Max(0, 100.0*(1.0-score/(255.0*255.0)))

	fmt.Printf("[Upload] Done. Score=%.2f, Similarity=%.1f%%\n", score, similarity)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":     "processed",
		"method":     method,
		"iterations": iterations,
		"similarity": similarity,
		"genome":     genome,
	})
}

// handleUndo pops the most recent genome from a cell's undo stack and
// reinstates it.
func handleUndo(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Index int `json:"index"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	if req.Index < 0 || req.Index >= totalCells {
		http.Error(w, "Invalid cell index", http.StatusBadRequest)
		return
	}
	state.mu.Lock()
	h := state.cells[req.Index].History
	if len(h) == 0 {
		state.mu.Unlock()
		http.Error(w, "Nothing to undo for this cell", http.StatusBadRequest)
		return
	}
	g := h[len(h)-1]
	state.cells[req.Index].History = h[:len(h)-1]
	state.cells[req.Index].Genome = g
	state.mu.Unlock()

	img := renderClean(g, imgW, imgH)
	state.mu.Lock()
	state.cells[req.Index].Image = imgToBase64(img)
	state.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "undone"})
}

// handleGetGenome serves a cell's FULL genome, luma reference included.
// The grid endpoint strips the reference to keep responses light; saving
// parameters needs it, so the client fetches it on demand.
func handleGetGenome(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Index int `json:"index"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	if req.Index < 0 || req.Index >= totalCells {
		http.Error(w, "Invalid cell index", http.StatusBadRequest)
		return
	}
	state.mu.RLock()
	g := state.cells[req.Index].Genome
	state.mu.RUnlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"genome": g})
}

// ============================================================================
// RENDERER FEATURE TOGGLES
// ============================================================================

// FeatureToggles controls which visual features freshly generated genomes
// may use. All default to on. Toggling affects only genomes created AFTER
// the change: applyFeatures zeroes disabled genes at creation time, so
// existing images and saved genomes render exactly as before.
type FeatureToggles struct {
	Transform  bool `json:"transform"`      // nonlinear transforms (wave 1)
	Relief     bool `json:"relief"`         // relief lighting (wave 1)
	Spikes     bool `json:"spikes"`         // spectral spikes (wave 2)
	Chroma     bool `json:"chroma"`         // chroma modulation (wave 2)
	Cone       bool `json:"cone"`           // directional cone (wave 3)
	DomainWarp bool `json:"domain_warp"`    // classic-mode warp (wave 4)
	Symmetry   bool `json:"symmetry"`       // radial symmetry (wave 5)
	AnchorPal  bool `json:"anchor_palette"` // anchor-point palette (wave 5)
}

var (
	featureMu sync.RWMutex
	features  = FeatureToggles{
		Transform:  true,
		Relief:     true,
		Spikes:     true,
		Chroma:     true,
		Cone:       true,
		DomainWarp: true,
		Symmetry:   true,
		AnchorPal:  true,
	}
)

func getFeatures() FeatureToggles {
	featureMu.RLock()
	defer featureMu.RUnlock()
	return features
}

func setFeatures(f FeatureToggles) {
	featureMu.Lock()
	features = f
	featureMu.Unlock()
}

// applyFeatures zeroes genes whose feature is disabled. Call it on every
// NEWLY created genome (random or bred) so unchecking affects only new
// images; the caller must not hold state.mu.
func applyFeatures(g *Genome) {
	f := getFeatures()
	if !f.Transform {
		g.Transform = 0
	}
	if !f.Relief {
		g.ReliefStrength = 0
	}
	if !f.Spikes {
		g.SpikeCount = 0
	}
	if !f.Chroma {
		g.ChromaStrength = 0
	}
	if !f.Cone {
		g.ConeWidth = 1.0
	}
	if !f.DomainWarp {
		g.DomainWarp = 0
	}
	if !f.Symmetry {
		g.SymmetryFold = 0
	}
	if !f.AnchorPal {
		g.PaletteMode = 0
	}
}

func handleFeatures(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	switch r.Method {
	case "GET":
		json.NewEncoder(w).Encode(getFeatures())
	case "POST":
		var f FeatureToggles
		if err := json.NewDecoder(r.Body).Decode(&f); err != nil {
			http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
			return
		}
		setFeatures(f)
		json.NewEncoder(w).Encode(getFeatures())
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// ============================================================================
// MAIN
// ============================================================================

func main() {
	genomeDir, _ := os.Getwd()
	os.MkdirAll(genomeDir, 0755)

	state = NewAppState(genomeDir)

	// Initialize grid with random images
	state.mu.Lock()
	for i := 0; i < totalCells; i++ {
		seedRng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(i)*1000000))
		genome := randomGenome(seedRng)
		applyFeatures(&genome)
		img := renderClean(genome, imgW, imgH)
		state.cells[i].Genome = genome
		state.cells[i].Image = imgToBase64(img)
	}
	state.mu.Unlock()

	// Routes
	http.HandleFunc("/", handleStatic)
	http.HandleFunc("/api/grid", handleGrid)
	http.HandleFunc("/api/evolve", handleEvolve)
	http.HandleFunc("/api/generate-all", handleGenerateAll)
	http.HandleFunc("/api/toggle-lock", handleToggleLock)
	http.HandleFunc("/api/save-image", handleSaveImage)
	http.HandleFunc("/api/save-params", handleSaveParams)
	http.HandleFunc("/api/load-params", handleLoadParams)
	http.HandleFunc("/api/upload-image", handleUploadImage)
	http.HandleFunc("/api/render-animation", handleRenderAnimation)
	http.HandleFunc("/api/render-image", handleRenderImage)
	http.HandleFunc("/api/undo", handleUndo)
	http.HandleFunc("/api/get-genome", handleGetGenome)
	http.HandleFunc("/api/features", handleFeatures)

	fmt.Printf("Genetic Image Evolution Lab\n")
	fmt.Printf("Server listening on http://localhost:%d\n", Port)
	fmt.Printf("Press Ctrl+C to stop\n")

	if err := http.ListenAndServe(fmt.Sprintf(":%d", Port), nil); err != nil {
		fmt.Println("Server error:", err)
	}
}
