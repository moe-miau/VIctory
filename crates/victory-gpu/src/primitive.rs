//! Console primitive types the host lacks. A rectangle list gives three
//! corners per rectangle, the second the one between the other two, and
//! the hardware adds the fourth at `v0 + v2 - v1` (Grand Theft Auto VI's
//! full-screen rectangles come as top left, bottom left, bottom right); on
//! the host each rectangle becomes two triangles, with the fourth vertex
//! made from the other three on the CPU.
//!
//! shortcut: the fourth vertex is computed from vertex inputs, not from
//! vertex shader outputs as the hardware does; that is the same for vertex
//! shaders linear in their inputs (blits, 2D quads). Inputs in formats
//! [`components`] does not know keep the first vertex's value, and so do
//! components past the end of the vertex (an element that reaches into the
//! next one, which the layout allows).

/// Console-only primitive topologies expanded into standard D3D12 triangle or
/// line list index buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpandedTopology {
    TriangleFan,
    LineLoop,
    Quads,
    QuadStrip,
}

impl ExpandedTopology {
    const TRIANGLE_FAN: u32 = 6;
    const LINE_LOOP: u32 = 18;
    const QUADS: u32 = 19;
    const QUAD_STRIP: u32 = 20;

    /// Returns the expanded console topology kind, or `None` for standard PC
    /// topologies and `RECTANGLES` (handled via vertex expansion).
    pub(crate) fn from_console(topology: u32) -> Option<Self> {
        match topology {
            Self::TRIANGLE_FAN => Some(Self::TriangleFan),
            Self::LINE_LOOP => Some(Self::LineLoop),
            Self::QUADS => Some(Self::Quads),
            Self::QUAD_STRIP => Some(Self::QuadStrip),
            _ => None,
        }
    }

    /// Standard D3D12 primitive topology value (`LINELIST = 2`, `TRIANGLELIST = 4`).
    pub(crate) fn host_topology(self) -> u32 {
        match self {
            Self::LineLoop => 2,
            Self::TriangleFan | Self::Quads | Self::QuadStrip => 4,
        }
    }

    /// Maximum number of indices emitted for `count` input vertices or indices.
    pub(crate) fn max_output_indices(self, count: u32) -> u32 {
        match self {
            Self::TriangleFan => count.saturating_sub(2).saturating_mul(3),
            Self::LineLoop => {
                if count >= 2 {
                    count.saturating_mul(2)
                } else {
                    0
                }
            }
            Self::Quads => (count / 4).saturating_mul(6),
            Self::QuadStrip => {
                if count >= 4 {
                    ((count - 2) / 2).saturating_mul(6)
                } else {
                    0
                }
            }
        }
    }

    /// Expands `count` sequential 16-bit vertex indices starting at `start`.
    pub(crate) fn expand_non_indexed_u16(self, count: u32, start: u16, dst: &mut [u16]) -> usize {
        expand_indices(
            self,
            (0..count).map(|i| start.wrapping_add(i as u16)),
            None,
            start,
            dst,
        )
    }

    /// Expands `count` sequential 32-bit vertex indices starting at `start`.
    pub(crate) fn expand_non_indexed_u32(self, count: u32, start: u32, dst: &mut [u32]) -> usize {
        expand_indices(
            self,
            (0..count).map(|i| start.wrapping_add(i)),
            None,
            start,
            dst,
        )
    }

    /// Expands 16-bit indices from `src`, honoring optional strip-cut `cut` and
    /// padding unused tail slots with degenerate primitives.
    pub(crate) fn expand_indexed_u16(
        self,
        src: &[u16],
        cut: Option<u16>,
        fallback_pad: u16,
        dst: &mut [u16],
    ) -> usize {
        expand_indices(self, src.iter().copied(), cut, fallback_pad, dst)
    }

    /// Expands 32-bit indices from `src`, honoring optional strip-cut `cut` and
    /// padding unused tail slots with degenerate primitives.
    pub(crate) fn expand_indexed_u32(
        self,
        src: &[u32],
        cut: Option<u32>,
        fallback_pad: u32,
        dst: &mut [u32],
    ) -> usize {
        expand_indices(self, src.iter().copied(), cut, fallback_pad, dst)
    }
}

fn expand_indices<T: Copy + Eq>(
    kind: ExpandedTopology,
    indices: impl IntoIterator<Item = T>,
    cut: Option<T>,
    fallback_pad: T,
    dst: &mut [T],
) -> usize {
    let mut written = 0usize;
    let mut emit = |slice: &[T]| {
        if written + slice.len() <= dst.len() {
            dst[written..written + slice.len()].copy_from_slice(slice);
            written += slice.len();
        }
    };
    match kind {
        ExpandedTopology::TriangleFan => {
            let mut pivot = None;
            let mut prev = None;
            for v in indices {
                if cut == Some(v) {
                    pivot = None;
                    prev = None;
                    continue;
                }
                match (pivot, prev) {
                    (None, _) => pivot = Some(v),
                    (Some(_), None) => prev = Some(v),
                    (Some(p), Some(q)) => {
                        emit(&[p, q, v]);
                        prev = Some(v);
                    }
                }
            }
        }
        ExpandedTopology::LineLoop => {
            let mut first = None;
            let mut prev = None;
            let mut run_len = 0usize;
            for v in indices {
                if cut == Some(v) {
                    if run_len >= 2
                        && let (Some(f), Some(p)) = (first, prev)
                    {
                        emit(&[p, f]);
                    }
                    first = None;
                    prev = None;
                    run_len = 0;
                    continue;
                }
                if let Some(p) = prev {
                    emit(&[p, v]);
                } else {
                    first = Some(v);
                }
                prev = Some(v);
                run_len += 1;
            }
            if run_len >= 2
                && let (Some(f), Some(p)) = (first, prev)
            {
                emit(&[p, f]);
            }
        }
        ExpandedTopology::Quads => {
            let mut q = [None; 3];
            let mut len = 0usize;
            for v in indices {
                if cut == Some(v) {
                    len = 0;
                    continue;
                }
                if len < 3 {
                    q[len] = Some(v);
                    len += 1;
                } else if let [Some(v0), Some(v1), Some(v2)] = q {
                    emit(&[v0, v1, v2, v0, v2, v]);
                    len = 0;
                }
            }
        }
        ExpandedTopology::QuadStrip => {
            let mut a = None;
            let mut b = None;
            let mut c = None;
            for v in indices {
                if cut == Some(v) {
                    a = None;
                    b = None;
                    c = None;
                    continue;
                }
                match (a, b, c) {
                    (None, _, _) => a = Some(v),
                    (Some(_), None, _) => b = Some(v),
                    (Some(_), Some(_), None) => c = Some(v),
                    (Some(v0), Some(v1), Some(v2)) => {
                        emit(&[v0, v1, v, v0, v, v2]);
                        a = Some(v2);
                        b = Some(v);
                        c = None;
                    }
                }
            }
        }
    }
    if written < dst.len() {
        let pad = if written > 0 { dst[0] } else { fallback_pad };
        dst[written..].fill(pad);
    }
    written
}

/// Expands `rects` rectangles of `stride`-byte vertices from `src` into six
/// vertices each in `dst`: `(v0, v1, v2)` and `(v0, v2, v3)`, so both
/// triangles turn the same way. `elements` are (byte offset, format).
pub(crate) fn expand_rects(src: &[u8], dst: &mut [u8], stride: usize, elements: &[(u32, u32)]) {
    for (rect, out) in src
        .chunks_exact(3 * stride)
        .zip(dst.chunks_exact_mut(6 * stride))
    {
        let v = |i: usize| &rect[i * stride..(i + 1) * stride];
        for (slot, i) in [0, 1, 2, 0, 2].into_iter().enumerate() {
            out[slot * stride..(slot + 1) * stride].copy_from_slice(v(i));
        }
        let fourth = &mut out[5 * stride..];
        fourth.copy_from_slice(v(0));
        for &(offset, format) in elements {
            let at = offset as usize;
            let Some((n, kind)) = components(format) else {
                continue;
            };
            // Only the components inside the vertex: the rest belong to the
            // next one.
            let inside = stride.saturating_sub(at) / kind.bytes();
            for c in 0..n.min(inside) {
                let get = |i: usize| kind.read(&v(i)[at..], c);
                let value = get(0) + get(2) - get(1);
                kind.write(&mut fourth[at..], c, value);
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Kind {
    F32,
    F16,
    Unorm8,
    Unorm16,
}

/// Component count and kind of the vertex formats rectangles use.
fn components(format: u32) -> Option<(usize, Kind)> {
    Some(match format {
        2 => (4, Kind::F32),
        6 => (3, Kind::F32),
        16 => (2, Kind::F32),
        41 => (1, Kind::F32),
        10 => (4, Kind::F16),
        34 => (2, Kind::F16),
        54 => (1, Kind::F16),
        11 => (4, Kind::Unorm16),
        35 => (2, Kind::Unorm16),
        28 | 29 | 87 | 91 => (4, Kind::Unorm8),
        _ => return None,
    })
}

impl Kind {
    fn bytes(self) -> usize {
        match self {
            Kind::F32 => 4,
            Kind::F16 | Kind::Unorm16 => 2,
            Kind::Unorm8 => 1,
        }
    }

    fn read(self, at: &[u8], c: usize) -> f32 {
        let b = &at[c * self.bytes()..];
        match self {
            Kind::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            Kind::F16 => half_to_f32(u16::from_le_bytes([b[0], b[1]])),
            Kind::Unorm16 => f32::from(u16::from_le_bytes([b[0], b[1]])) / 65535.0,
            Kind::Unorm8 => f32::from(b[0]) / 255.0,
        }
    }

    fn write(self, at: &mut [u8], c: usize, v: f32) {
        let n = self.bytes();
        let b = &mut at[c * n..(c + 1) * n];
        match self {
            Kind::F32 => b.copy_from_slice(&v.to_le_bytes()),
            Kind::F16 => b.copy_from_slice(&f32_to_half(v).to_le_bytes()),
            Kind::Unorm16 => {
                b.copy_from_slice(&((v.clamp(0.0, 1.0) * 65535.0).round() as u16).to_le_bytes())
            }
            Kind::Unorm8 => b[0] = (v.clamp(0.0, 1.0) * 255.0).round() as u8,
        }
    }
}

fn half_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        // Subnormal: scale the mantissa.
        (0, m) => {
            let v = m as f32 * 2f32.powi(-24);
            return if sign != 0 { -v } else { v };
        }
        (31, m) => sign | 0x7f80_0000 | m << 13,
        (e, m) => sign | (e + 112) << 23 | m << 13,
    };
    f32::from_bits(bits)
}

fn f32_to_half(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let man = bits & 0x7f_ffff;
    if exp >= 31 {
        return sign | 0x7c00; // too large: infinity
    }
    if exp <= 0 {
        if exp < -10 {
            return sign; // below half the smallest subnormal: zero
        }
        // Half subnormals are `m * 2^-24`. An f32 in this range is still
        // normal (exponent field >= 1), so its significand has the implicit
        // leading 1; shift it into the half mantissa, rounding to even.
        let sig = (1u32 << 23) | man;
        let shift = (14 - exp) as u32;
        let halfway = 1u32 << (shift - 1);
        let q = sig >> shift;
        let r = sig & ((1u32 << shift) - 1);
        let m = q + u32::from(r > halfway || (r == halfway && (q & 1) != 0));
        return sign | m as u16;
    }
    // Rounded to nearest: a carry out of the mantissa is added into the
    // exponent (`2 - 2^-12` is 2.0); at most it reaches infinity, 0x7c00.
    let half = (exp as u32) << 10;
    sign | (half + ((man + 0x1000) >> 13)) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rectangle of float2 positions and RGBA8 colors, given as top
    /// left, bottom left, bottom right: the fourth corner completes the
    /// parallelogram at the top right.
    #[test]
    fn rectangles_get_their_fourth_corner() {
        let vertex = |x: f32, y: f32, r: u8| {
            let mut v = Vec::new();
            v.extend(x.to_le_bytes());
            v.extend(y.to_le_bytes());
            v.extend([r, 0, 0, 255]);
            v
        };
        let src: Vec<u8> = [
            vertex(-1.0, 1.0, 0),
            vertex(-1.0, -1.0, 100),
            vertex(1.0, -1.0, 150),
        ]
        .concat();
        let mut dst = vec![0u8; 6 * 12];
        expand_rects(&src, &mut dst, 12, &[(0, 16), (8, 28)]);
        assert_eq!(dst[..36], src[..]);
        assert_eq!(dst[36..48], src[..12]);
        assert_eq!(dst[48..60], src[24..36]);
        assert_eq!(dst[60..], vertex(1.0, 1.0, 50)[..]);
        assert_eq!(f32_to_half(1.5), 0x3e00);
        assert_eq!(half_to_f32(0x3e00), 1.5);
        assert_eq!(half_to_f32(0xc000), -2.0);
    }

    /// A rounding that carries out of the mantissa raises the exponent
    /// (the carry used to be or-ed into an odd exponent, halving the value).
    #[test]
    fn halves_round_up_into_the_next_power_of_two() {
        assert_eq!(f32_to_half(2.0 - 2f32.powi(-12)), 0x4000);
        assert_eq!(f32_to_half(0.5 - 2f32.powi(-13)), 0x3800);
        assert_eq!(f32_to_half(65520.0), 0x7c00, "past the largest half");
        // An R16_FLOAT element at 1.0, 2^-12 and 1.0: the fourth corner is
        // 2 - 2^-12, which rounds to 2.0.
        let src: Vec<u8> = [0x3c00_u16, 0x0c00, 0x3c00]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        let mut dst = vec![0u8; 6 * 2];
        expand_rects(&src, &mut dst, 2, &[(0, 54)]);
        assert_eq!(dst[10..], 0x4000_u16.to_le_bytes());
    }

    /// An element that reaches past the stride (float4 at 4 in 8-byte
    /// vertices, RGBA8 at the stride itself): the components inside get
    /// the fourth corner, the rest keep the first vertex's bytes. The old
    /// code indexed past the vertex and panicked.
    #[test]
    fn elements_past_the_stride_are_left_alone() {
        let vertex = |x: f32, y: f32| [x.to_le_bytes(), y.to_le_bytes()].concat();
        let src = [vertex(0.0, 2.0), vertex(1.0, 3.0), vertex(5.0, 7.0)].concat();
        let mut dst = vec![0u8; 6 * 8];
        expand_rects(&src, &mut dst, 8, &[(0, 16), (4, 2), (8, 28)]);
        assert_eq!(dst[..40], [&src[..], &src[..8], &src[16..]].concat()[..]);
        assert_eq!(dst[40..], vertex(4.0, 6.0)[..]);
    }

    /// `half_to_f32` reads subnormals, so an f16 rectangle output must be
    /// able to write them back instead of flushing every one to zero.
    #[test]
    fn half_subnormals_convert_in_both_directions() {
        assert_eq!(f32_to_half(2f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_half(-2f32.powi(-24)), 0x8001);
        assert_eq!(f32_to_half(2f32.powi(-23)), 0x0002);
        assert_eq!(f32_to_half(1023.0 * 2f32.powi(-24)), 0x03ff);
        // Ties go to an even mantissa: 0/1 and 1/2.
        assert_eq!(f32_to_half(2f32.powi(-25)), 0x0000);
        assert_eq!(f32_to_half(3.0 * 2f32.powi(-25)), 0x0002);
        // Below half the smallest subnormal, and normal values, are unchanged.
        assert_eq!(f32_to_half(2f32.powi(-26)), 0x0000);
        assert_eq!(f32_to_half(1.5), 0x3e00);
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
    }

    #[test]
    fn expand_non_indexed_topologies() {
        // QUADS: 2 quads + 1 trailing vertex ignored.
        let mut out = vec![0u32; ExpandedTopology::Quads.max_output_indices(9) as usize];
        assert_eq!(out.len(), 12);
        let n = ExpandedTopology::Quads.expand_non_indexed_u32(9, 10, &mut out);
        assert_eq!(n, 12);
        assert_eq!(out, [10, 11, 12, 10, 12, 13, 14, 15, 16, 14, 16, 17]);

        // QUAD_STRIP: 6 vertices -> 2 quads with consistent winding, 7th ignored.
        let mut out = vec![0u16; ExpandedTopology::QuadStrip.max_output_indices(7) as usize];
        assert_eq!(out.len(), 12);
        let n = ExpandedTopology::QuadStrip.expand_non_indexed_u16(7, 0, &mut out);
        assert_eq!(n, 12);
        assert_eq!(out, [0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4]);

        // TRIANGLE_FAN: 5 vertices -> 3 triangles sharing pivot 0.
        let mut out = vec![0u32; ExpandedTopology::TriangleFan.max_output_indices(5) as usize];
        assert_eq!(out.len(), 9);
        assert_eq!(
            ExpandedTopology::TriangleFan.expand_non_indexed_u32(5, 0, &mut out),
            9
        );
        assert_eq!(out, [0, 1, 2, 0, 2, 3, 0, 3, 4]);
        assert_eq!(ExpandedTopology::TriangleFan.max_output_indices(2), 0);

        // LINE_LOOP: 4 vertices -> 4 segments closing 3 -> 0.
        let mut out = vec![0u16; ExpandedTopology::LineLoop.max_output_indices(4) as usize];
        assert_eq!(out.len(), 8);
        assert_eq!(
            ExpandedTopology::LineLoop.expand_non_indexed_u16(4, 1, &mut out),
            8
        );
        assert_eq!(out, [1, 2, 2, 3, 3, 4, 4, 1]);
        assert_eq!(ExpandedTopology::LineLoop.max_output_indices(1), 0);
    }

    #[test]
    fn expand_indexed_topologies_with_strip_cuts() {
        // TRIANGLE_FAN (16-bit, 0xFFFF cut): two fans separated by cut + incomplete tail.
        let src: [u16; 10] = [0, 1, 2, 3, u16::MAX, 10, 11, 12, u16::MAX, 20];
        let mut out = vec![0u16; ExpandedTopology::TriangleFan.max_output_indices(10) as usize];
        let n = ExpandedTopology::TriangleFan.expand_indexed_u16(&src, Some(u16::MAX), 0, &mut out);
        assert_eq!(n, 9);
        assert_eq!(&out[..9], [0, 1, 2, 0, 2, 3, 10, 11, 12]);
        assert!(out[9..].iter().all(|&i| i == 0), "degenerate tail padding");

        // QUAD_STRIP (32-bit, 0xFFFFFFFF cut): two 4-vertex strips separated by cut.
        let src32: [u32; 9] = [0, 1, 2, 3, u32::MAX, 4, 5, 6, 7];
        let mut out32 = vec![0u32; ExpandedTopology::QuadStrip.max_output_indices(9) as usize];
        let n =
            ExpandedTopology::QuadStrip.expand_indexed_u32(&src32, Some(u32::MAX), 0, &mut out32);
        assert_eq!(n, 12);
        assert_eq!(&out32[..12], [0, 1, 3, 0, 3, 2, 4, 5, 7, 4, 7, 6]);
        assert!(out32[12..].iter().all(|&i| i == 0));

        // QUADS (16-bit cut discards incomplete 2-vertex quad before cut).
        let qsrc: [u16; 7] = [0, 1, u16::MAX, 2, 3, 4, 5];
        let mut qout = vec![0u16; ExpandedTopology::Quads.max_output_indices(7) as usize];
        let n = ExpandedTopology::Quads.expand_indexed_u16(&qsrc, Some(u16::MAX), 0, &mut qout);
        assert_eq!(n, 6);
        assert_eq!(qout, [2, 3, 4, 2, 4, 5]);

        // LINE_LOOP (16-bit cut closes first loop before starting second).
        let lsrc: [u16; 7] = [0, 1, 2, u16::MAX, 5, 6, 7];
        let mut lout = vec![0u16; ExpandedTopology::LineLoop.max_output_indices(7) as usize];
        let n = ExpandedTopology::LineLoop.expand_indexed_u16(&lsrc, Some(u16::MAX), 0, &mut lout);
        assert_eq!(n, 12);
        assert_eq!(&lout[..12], [0, 1, 1, 2, 2, 0, 5, 6, 6, 7, 7, 5]);
        assert_eq!(&lout[12..], [0, 0]);
    }

    #[cfg(windows)]
    fn render_topology_test(
        topology_type: windows::Win32::Graphics::Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE,
        host_topology: u32,
        strip_cut: windows::Win32::Graphics::Direct3D12::D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
        vertices: &[[f32; 2]],
        draw_calls: &[(&[u16], i32)],
    ) -> Option<Vec<[u8; 4]>> {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY, ID3DBlob};
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::core::s;

        let host = crate::host::host()?;
        const HLSL: &str = r#"
float4 vs_main(float2 pos : POSITION) : SV_Position { return float4(pos, 0.0, 1.0); }
float4 ps_main() : SV_Target { return float4(1.0, 1.0, 1.0, 1.0); }
"#;
        let compile = |entry: windows::core::PCSTR, target: windows::core::PCSTR| -> Vec<u8> {
            let mut blob: Option<ID3DBlob> = None;
            // SAFETY: static shader source and entry point strings.
            unsafe {
                D3DCompile(
                    HLSL.as_ptr().cast(),
                    HLSL.len(),
                    s!("topology_test"),
                    None,
                    None,
                    entry,
                    target,
                    0,
                    0,
                    &mut blob,
                    None,
                )
                .unwrap();
                let blob = blob.unwrap();
                std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                )
                .to_vec()
            }
        };
        let vs = compile(s!("vs_main"), s!("vs_5_0"));
        let ps = compile(s!("ps_main"), s!("ps_5_0"));

        let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
            Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
            ..Default::default()
        };
        let mut rs_blob: Option<ID3DBlob> = None;
        // SAFETY: plain root signature, PSO, RT, and readback creation.
        let (root, pso, rt, readback, vb, vb_gpu) = unsafe {
            D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut rs_blob, None)
                .unwrap();
            let rs_blob = rs_blob.unwrap();
            let rs_bytes = std::slice::from_raw_parts(
                rs_blob.GetBufferPointer().cast::<u8>(),
                rs_blob.GetBufferSize(),
            );
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, rs_bytes).unwrap();
            let elem = D3D12_INPUT_ELEMENT_DESC {
                SemanticName: s!("POSITION"),
                Format: DXGI_FORMAT_R32G32_FLOAT,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                ..Default::default()
            };
            let mut blend = D3D12_BLEND_DESC::default();
            blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            let mut rast = D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_BACK,
                DepthClipEnable: true.into(),
                ..Default::default()
            };
            if topology_type == D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE {
                rast.CullMode = D3D12_CULL_MODE_NONE;
            }
            let mut pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: vs.as_ptr().cast(),
                    BytecodeLength: vs.len(),
                },
                PS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: ps.as_ptr().cast(),
                    BytecodeLength: ps.len(),
                },
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: rast,
                InputLayout: D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: &elem,
                    NumElements: 1,
                },
                IBStripCutValue: strip_cut,
                PrimitiveTopologyType: topology_type,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            let pso: ID3D12PipelineState =
                host.device.CreateGraphicsPipelineState(&pso_desc).unwrap();

            let rt_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 8,
                Height: 8,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let def_heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let mut rt: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &def_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &rt_desc,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut rt,
                )
                .unwrap();
            let rt = rt.unwrap();

            let rb_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256 * 8,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let rb_heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let mut readback: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &rb_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &rb_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
            let readback = readback.unwrap();

            let vb_bytes = std::mem::size_of_val(vertices) as u64;
            let (vb, vb_ptr) = host.upload_buffer(vb_bytes).unwrap();
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr().cast::<u8>(),
                vb_ptr,
                vb_bytes as usize,
            );
            let vb_gpu = vb.GetGPUVirtualAddress();
            (root, pso, rt, readback, vb, vb_gpu)
        };

        let list = host.new_list().unwrap();
        let rtv = host.rtv(&rt, None);
        // SAFETY: open command list, live resources, valid scratch index views.
        unsafe {
            let l = &list.list;
            l.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 1.0], None);
            l.OMSetRenderTargets(1, Some(&rtv), false, None);
            l.RSSetViewports(&[D3D12_VIEWPORT {
                Width: 8.0,
                Height: 8.0,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            l.RSSetScissorRects(&[RECT {
                right: 8,
                bottom: 8,
                ..Default::default()
            }]);
            l.SetGraphicsRootSignature(&root);
            l.SetPipelineState(&pso);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host_topology as i32));
            l.IASetVertexBuffers(
                0,
                Some(&[D3D12_VERTEX_BUFFER_VIEW {
                    BufferLocation: vb_gpu,
                    SizeInBytes: std::mem::size_of_val(vertices) as u32,
                    StrideInBytes: 8,
                }]),
            );
            for &(indices, base) in draw_calls {
                let bytes = (indices.len() * 2) as u64;
                let (ib_gpu, ib_ptr) = list.scratch(bytes).unwrap();
                std::ptr::copy_nonoverlapping(
                    indices.as_ptr().cast::<u8>(),
                    ib_ptr,
                    bytes as usize,
                );
                l.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW {
                    BufferLocation: ib_gpu,
                    SizeInBytes: bytes as u32,
                    Format: DXGI_FORMAT_R16_UINT,
                }));
                l.DrawIndexedInstanced(indices.len() as u32, 1, 0, base, 0);
            }
            l.ResourceBarrier(&[crate::host::transition(
                &rt,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                            Width: 8,
                            Height: 8,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&rt),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        list.close();
        host.execute(&[&list]);
        drop(vb);

        let mut data = std::ptr::null_mut();
        // SAFETY: `host.execute` waited for the GPU copy into `readback`.
        let pixels = unsafe {
            readback.Map(0, None, Some(&mut data)).unwrap();
            let mut out = Vec::with_capacity(64);
            for y in 0..8usize {
                let row = std::slice::from_raw_parts(data.cast::<u8>().add(y * 256), 32);
                for x in 0..8usize {
                    out.push([row[x * 4], row[x * 4 + 1], row[x * 4 + 2], row[x * 4 + 3]]);
                }
            }
            readback.Unmap(0, None);
            out
        };
        Some(pixels)
    }

    #[cfg(windows)]
    #[test]
    fn quads_and_strip_cut_triangle_fan_render_with_backface_culling() {
        use windows::Win32::Graphics::Direct3D12::*;
        // Vertices in clockwise NDC order (front-facing under D3D12_CULL_MODE_BACK):
        // 0..4: non-indexed QUADS covering left half (x: -1..0, y: -1..1).
        // 4..8: indexed TRIANGLE_FAN #1 covering top-right (x: 0.25..1, y: 0.25..1).
        // 8..11: indexed TRIANGLE_FAN #2 covering bottom-right (x: 0.25..1, y: -1..-0.25).
        let vertices: [[f32; 2]; 11] = [
            [-1.0, 1.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [-1.0, -1.0],
            [0.25, 1.0],
            [1.0, 1.0],
            [1.0, 0.25],
            [0.25, 0.25],
            [0.25, -0.25],
            [1.0, -0.25],
            [1.0, -1.0],
        ];
        let mut quad_ib = vec![0u16; ExpandedTopology::Quads.max_output_indices(4) as usize];
        ExpandedTopology::Quads.expand_non_indexed_u16(4, 0, &mut quad_ib);

        let fan_src: [u16; 8] = [4, 5, 6, 7, u16::MAX, 8, 9, 10];
        let mut fan_ib = vec![0u16; ExpandedTopology::TriangleFan.max_output_indices(8) as usize];
        ExpandedTopology::TriangleFan.expand_indexed_u16(&fan_src, Some(u16::MAX), 0, &mut fan_ib);

        let Some(px) = render_topology_test(
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            ExpandedTopology::Quads.host_topology(),
            D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF,
            &vertices,
            &[(&quad_ib, 0), (&fan_ib, 0)],
        ) else {
            return;
        };
        let at = |x: usize, y: usize| px[y * 8 + x][0];
        // Both triangles of the left-half quad must rasterize (not backface-culled).
        assert_eq!(at(1, 1), 255, "quad first triangle");
        assert_eq!(at(1, 6), 255, "quad second triangle");
        // Both strip-cut-separated triangle fans on the right half must rasterize.
        assert_eq!(at(6, 1), 255, "first triangle fan");
        assert_eq!(at(7, 6), 255, "second triangle fan after strip cut");
        // Middle gap between the two fans must stay black (strip cut did not bridge).
        assert_eq!(at(6, 4), 0, "gap between strip-cut fans stays black");
    }

    #[cfg(windows)]
    #[test]
    fn quad_strip_and_line_loop_render_on_gpu() {
        use windows::Win32::Graphics::Direct3D12::*;
        // Indexed QUAD_STRIP with 6 vertices (2 quads, CW front-facing under CULL_MODE_BACK)
        // covering the entire 8x8 target.
        let strip_verts: [[f32; 2]; 6] = [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [0.0, -1.0],
            [0.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ];
        let strip_src: [u16; 6] = [0, 1, 2, 3, 4, 5];
        let mut strip_ib = vec![0u16; ExpandedTopology::QuadStrip.max_output_indices(6) as usize];
        ExpandedTopology::QuadStrip.expand_indexed_u16(&strip_src, None, 0, &mut strip_ib);

        let Some(px) = render_topology_test(
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            ExpandedTopology::QuadStrip.host_topology(),
            D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            &strip_verts,
            &[(&strip_ib, 0)],
        ) else {
            return;
        };
        let pat = |x: usize, y: usize| px[y * 8 + x][0];
        assert_eq!(pat(1, 1), 255, "quad strip quad 0");
        assert_eq!(pat(6, 6), 255, "quad strip quad 1");

        // LINE_LOOP drawing the perimeter of pixel centers (1,1)->(6,1)->(6,6)->(1,6)->(1,1).
        let to_ndc = |p: f32| (p + 0.5) / 8.0 * 2.0 - 1.0;
        let loop_verts: [[f32; 2]; 4] = [
            [to_ndc(1.0), -to_ndc(1.0)],
            [to_ndc(6.0), -to_ndc(1.0)],
            [to_ndc(6.0), -to_ndc(6.0)],
            [to_ndc(1.0), -to_ndc(6.0)],
        ];
        let mut loop_ib = vec![0u16; ExpandedTopology::LineLoop.max_output_indices(4) as usize];
        ExpandedTopology::LineLoop.expand_non_indexed_u16(4, 0, &mut loop_ib);

        let Some(lpx) = render_topology_test(
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
            ExpandedTopology::LineLoop.host_topology(),
            D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            &loop_verts,
            &[(&loop_ib, 0)],
        ) else {
            return;
        };
        let lat = |x: usize, y: usize| lpx[y * 8 + x][0];
        assert_eq!(lat(3, 1), 255, "top edge");
        assert_eq!(lat(6, 3), 255, "right edge");
        assert_eq!(lat(3, 6), 255, "bottom edge");
        assert_eq!(lat(1, 3), 255, "closing edge v3 -> v0");
        assert_eq!(lat(3, 3), 0, "interior stays black");
    }
}
