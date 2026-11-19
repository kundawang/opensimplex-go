package opensimplex

import (
	"math"
	"testing"
)

// warpGrid is a fixed set of sample points used across the warped-sampler
// tests. All coordinates are exactly representable in both float32 and
// float64 so the two interfaces can be compared point for point.
func warpGrid() [][2]float64 {
	var pts [][2]float64
	for i := -8; i <= 8; i++ {
		for j := -8; j <= 8; j++ {
			pts = append(pts, [2]float64{float64(i) * 0.5, float64(j) * 0.5})
		}
	}
	return pts
}

func mustWarped(t *testing.T, cfg WarpConfig) WarpedNoise {
	t.Helper()
	w, err := NewWarped(cfg)
	if err != nil {
		t.Fatalf("NewWarped(%+v) returned unexpected error: %v", cfg, err)
	}
	return w
}

// Zero warp strength must degenerate to plain fractal noise: the same
// octaves evaluated at the undisplaced coordinates, normalized by the
// amplitude sum.
func TestWarpedZeroWarpMatchesPlainFractal(t *testing.T) {
	const seed = 12345
	cfg := WarpConfig{
		Seed:         seed,
		Octaves:      4,
		Frequency:    1.5,
		Lacunarity:   2,
		Gain:         0.5,
		WarpStrength: 0,
	}
	w := mustWarped(t, cfg)
	base := New(seed)

	for _, p := range warpGrid() {
		sum := 0.0
		norm := 0.0
		amp := 1.0
		freq := cfg.Frequency
		for i := 0; i < cfg.Octaves; i++ {
			sum += amp * base.Eval2(p[0]*freq, p[1]*freq)
			norm += amp
			amp *= cfg.Gain
			freq *= cfg.Lacunarity
		}
		expected := sum / norm
		if got := w.Eval2(p[0], p[1]); got != expected {
			t.Fatalf("zero-warp Eval2(%v, %v) = %v, want plain fractal %v",
				p[0], p[1], got, expected)
		}
	}
}

// A single-octave, zero-warp sampler with unit frequency must reproduce the
// underlying OpenSimplex field exactly.
func TestWarpedSingleOctaveMatchesBaseNoise(t *testing.T) {
	const seed = 777
	cfg := DefaultWarpConfig(seed)
	cfg.Octaves = 1
	cfg.Frequency = 1
	cfg.WarpStrength = 0
	w := mustWarped(t, cfg)
	base := New(seed)

	for _, p := range warpGrid() {
		if got, want := w.Eval2(p[0], p[1]), base.Eval2(p[0], p[1]); got != want {
			t.Fatalf("Eval2(%v, %v) = %v, want base noise %v", p[0], p[1], got, want)
		}
	}
}

// Equal configs must be point-for-point deterministic; different seeds must
// diverge.
func TestWarpedSeedDeterminismAndDifference(t *testing.T) {
	cfg := DefaultWarpConfig(42)
	a := mustWarped(t, cfg)
	b := mustWarped(t, cfg)

	other := DefaultWarpConfig(43)
	c := mustWarped(t, other)

	differ := 0
	for _, p := range warpGrid() {
		if va, vb := a.Eval2(p[0], p[1]), b.Eval2(p[0], p[1]); va != vb {
			t.Fatalf("same seed diverged at (%v, %v): %v vs %v", p[0], p[1], va, vb)
		}
		if a.Eval2(p[0], p[1]) != c.Eval2(p[0], p[1]) {
			differ++
		}
	}
	if differ == 0 {
		t.Fatal("different seeds produced identical output on the whole grid")
	}
}

// Neighbouring samples must stay close: the warped field is continuous, so a
// small step cannot produce a jump.
func TestWarpedContinuity(t *testing.T) {
	cfg := DefaultWarpConfig(9)
	cfg.WarpStrength = 2
	w := mustWarped(t, cfg)

	const step = 1e-3
	const maxJump = 0.1
	for _, p := range warpGrid() {
		v := w.Eval2(p[0], p[1])
		if d := math.Abs(w.Eval2(p[0]+step, p[1]) - v); d > maxJump {
			t.Fatalf("x-discontinuity at (%v, %v): jump %v", p[0], p[1], d)
		}
		if d := math.Abs(w.Eval2(p[0], p[1]+step) - v); d > maxJump {
			t.Fatalf("y-discontinuity at (%v, %v): jump %v", p[0], p[1], d)
		}
	}
}

// Output must stay inside the normalized [-1, 1] range, including under
// heavy warping.
func TestWarpedOutputRange(t *testing.T) {
	for _, strength := range []float64{0, 0.5, 8, 100} {
		cfg := DefaultWarpConfig(2024)
		cfg.WarpStrength = strength
		w := mustWarped(t, cfg)
		for _, p := range warpGrid() {
			v := w.Eval2(p[0], p[1])
			if math.IsNaN(v) || math.IsInf(v, 0) || v < -1 || v > 1 {
				t.Fatalf("strength %v: Eval2(%v, %v) = %v outside [-1, 1]",
					strength, p[0], p[1], v)
			}
		}
	}
}

func TestWarpedInvalidConfigs(t *testing.T) {
	valid := DefaultWarpConfig(1)

	bad := []WarpConfig{
		// Octave count out of range.
		func() WarpConfig { c := valid; c.Octaves = 0; return c }(),
		func() WarpConfig { c := valid; c.Octaves = -3; return c }(),
		func() WarpConfig { c := valid; c.Octaves = MaxWarpOctaves + 1; return c }(),
		// Zero/negative/non-finite frequency.
		func() WarpConfig { c := valid; c.Frequency = 0; return c }(),
		func() WarpConfig { c := valid; c.Frequency = -2; return c }(),
		func() WarpConfig { c := valid; c.Frequency = math.NaN(); return c }(),
		func() WarpConfig { c := valid; c.Frequency = math.Inf(1); return c }(),
		// Invalid frequency multiplier.
		func() WarpConfig { c := valid; c.Lacunarity = 0; return c }(),
		func() WarpConfig { c := valid; c.Lacunarity = 0.5; return c }(),
		func() WarpConfig { c := valid; c.Lacunarity = math.NaN(); return c }(),
		func() WarpConfig { c := valid; c.Lacunarity = math.Inf(1); return c }(),
		// Invalid amplitude decay.
		func() WarpConfig { c := valid; c.Gain = 0; return c }(),
		func() WarpConfig { c := valid; c.Gain = -0.5; return c }(),
		func() WarpConfig { c := valid; c.Gain = 1.5; return c }(),
		func() WarpConfig { c := valid; c.Gain = math.NaN(); return c }(),
		// Negative/non-finite warp strength.
		func() WarpConfig { c := valid; c.WarpStrength = -1; return c }(),
		func() WarpConfig { c := valid; c.WarpStrength = math.NaN(); return c }(),
		func() WarpConfig { c := valid; c.WarpStrength = math.Inf(1); return c }(),
		// Effective frequency of the highest octave beyond the cap.
		func() WarpConfig { c := valid; c.Frequency = 1 << 19; c.Octaves = 4; return c }(),
		func() WarpConfig { c := valid; c.Lacunarity = 1e6; return c }(),
	}
	for i, cfg := range bad {
		if _, err := NewWarped(cfg); err == nil {
			t.Errorf("bad config %d (%+v) was accepted", i, cfg)
		}
		if _, err := NewWarped32(cfg); err == nil {
			t.Errorf("bad config %d (%+v) was accepted by NewWarped32", i, cfg)
		}
	}

	good := []WarpConfig{
		valid,
		func() WarpConfig { c := valid; c.Octaves = 1; return c }(),
		func() WarpConfig { c := valid; c.Octaves = MaxWarpOctaves; c.Lacunarity = 1; return c }(),
		func() WarpConfig { c := valid; c.Lacunarity = 1; return c }(),
		func() WarpConfig { c := valid; c.Gain = 1; return c }(),
		func() WarpConfig { c := valid; c.WarpStrength = 0; return c }(),
		func() WarpConfig { c := valid; c.Octaves = 1; c.Frequency = 1 << 20; return c }(),
	}
	for i, cfg := range good {
		if _, err := NewWarped(cfg); err != nil {
			t.Errorf("good config %d (%+v) rejected: %v", i, cfg, err)
		}
	}
}

// Non-finite coordinates must yield the defined neutral value 0, not NaN.
func TestWarpedNonFiniteCoordinates(t *testing.T) {
	w := mustWarped(t, DefaultWarpConfig(5))
	bad := [][2]float64{
		{math.NaN(), 0},
		{0, math.NaN()},
		{math.Inf(1), 0},
		{0, math.Inf(-1)},
		{math.Inf(1), math.Inf(1)},
	}
	for _, p := range bad {
		if v := w.Eval2(p[0], p[1]); v != 0 {
			t.Errorf("Eval2(%v, %v) = %v, want 0", p[0], p[1], v)
		}
	}

	w32, err := NewWarped32(DefaultWarpConfig(5))
	if err != nil {
		t.Fatal(err)
	}
	if v := w32.Eval2(float32(math.NaN()), 0); v != 0 {
		t.Errorf("WarpedNoise32.Eval2(NaN, 0) = %v, want 0", v)
	}
}

// Extremely large finite coordinates and warp strengths must still produce
// finite, bounded results.
func TestWarpedExtremeCoordinates(t *testing.T) {
	cfg := DefaultWarpConfig(11)
	cfg.WarpStrength = 1e300 // finite, so it is accepted
	w := mustWarped(t, cfg)

	coords := []float64{1e15, -1e15, 1e300, -1e300, math.MaxFloat64, -math.MaxFloat64}
	for _, x := range coords {
		for _, y := range coords {
			v := w.Eval2(x, y)
			if math.IsNaN(v) || math.IsInf(v, 0) || v < -1 || v > 1 {
				t.Fatalf("Eval2(%v, %v) = %v, want finite value in [-1, 1]", x, y, v)
			}
		}
	}
}

// The float32 interface must share the float64 semantics: identical results
// (up to the float32 cast) on exactly-representable points, and the same
// behaviour for non-finite input.
func TestWarped32Consistency(t *testing.T) {
	cfg := DefaultWarpConfig(31415)
	w64 := mustWarped(t, cfg)
	w32, err := NewWarped32(cfg)
	if err != nil {
		t.Fatal(err)
	}

	for _, p := range warpGrid() {
		x, y := float32(p[0]), float32(p[1])
		got := w32.Eval2(x, y)
		want := float32(w64.Eval2(float64(x), float64(y)))
		if got != want {
			t.Fatalf("32/64 mismatch at (%v, %v): %v vs %v", x, y, got, want)
		}
		if got < -1 || got > 1 {
			t.Fatalf("float32 result %v outside [-1, 1] at (%v, %v)", got, x, y)
		}
	}
}
