#include "stdafx.h"
#include <xmmintrin.h>
#include <immintrin.h>

void VSFastMulSIMD(const Matrix3x3W & InM1, const Matrix3x3W & InM2, Matrix3x3W & OutM)
{
    // unaligned memory ? row major ?
    __m128 m11 = _mm_loadu_ps((const float*)&InM1);
    __m128 m12 = _mm_loadu_ps((const float*)&InM1 + 4);
    __m128 m13 = _mm_loadu_ps((const float*)&InM1 + 8);
    __m128 m14 = _mm_loadu_ps((const float*)&InM1 + 12);

    __m128 m21 = _mm_loadu_ps((const float*)&InM2);
    __m128 m22 = _mm_loadu_ps((const float*)&InM2 + 4);
    __m128 m23 = _mm_loadu_ps((const float*)&InM2 + 8);
    __m128 m24 = _mm_loadu_ps((const float*)&InM2 + 12);

    {
        __m128 e0 = _mm_shuffle_ps(m21, m21, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 e1 = _mm_shuffle_ps(m21, m21, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 e2 = _mm_shuffle_ps(m21, m21, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 e3 = _mm_shuffle_ps(m21, m21, _MM_SHUFFLE(3, 3, 3, 3));

        __m128 m0 = _mm_mul_ps(m11, e0);
        __m128 m1 = _mm_mul_ps(m12, e1);
        __m128 m2 = _mm_mul_ps(m13, e2);
        __m128 m3 = _mm_mul_ps(m14, e3);

        __m128 a0 = _mm_add_ps(m0, m1);
        __m128 a1 = _mm_add_ps(m2, m3);
        __m128 a2 = _mm_add_ps(a0, a1);

        _mm_storeu_ps(OutM.M[0], a2);
    }

    {
        __m128 e0 = _mm_shuffle_ps(m22, m22, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 e1 = _mm_shuffle_ps(m22, m22, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 e2 = _mm_shuffle_ps(m22, m22, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 e3 = _mm_shuffle_ps(m22, m22, _MM_SHUFFLE(3, 3, 3, 3));

        __m128 m0 = _mm_mul_ps(m11, e0);
        __m128 m1 = _mm_mul_ps(m12, e1);
        __m128 m2 = _mm_mul_ps(m13, e2);
        __m128 m3 = _mm_mul_ps(m14, e3);

        __m128 a0 = _mm_add_ps(m0, m1);
        __m128 a1 = _mm_add_ps(m2, m3);
        __m128 a2 = _mm_add_ps(a0, a1);

        _mm_storeu_ps(OutM.M[1], a2);
    }

    {
        __m128 e0 = _mm_shuffle_ps(m23, m23, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 e1 = _mm_shuffle_ps(m23, m23, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 e2 = _mm_shuffle_ps(m23, m23, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 e3 = _mm_shuffle_ps(m23, m23, _MM_SHUFFLE(3, 3, 3, 3));

        __m128 m0 = _mm_mul_ps(m11, e0);
        __m128 m1 = _mm_mul_ps(m12, e1);
        __m128 m2 = _mm_mul_ps(m13, e2);
        __m128 m3 = _mm_mul_ps(m14, e3);

        __m128 a0 = _mm_add_ps(m0, m1);
        __m128 a1 = _mm_add_ps(m2, m3);
        __m128 a2 = _mm_add_ps(a0, a1);

        _mm_storeu_ps(OutM.M[2], a2);
    }

    {
        __m128 e0 = _mm_shuffle_ps(m24, m24, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 e1 = _mm_shuffle_ps(m24, m24, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 e2 = _mm_shuffle_ps(m24, m24, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 e3 = _mm_shuffle_ps(m24, m24, _MM_SHUFFLE(3, 3, 3, 3));

        __m128 m0 = _mm_mul_ps(m11, e0);
        __m128 m1 = _mm_mul_ps(m12, e1);
        __m128 m2 = _mm_mul_ps(m13, e2);
        __m128 m3 = _mm_mul_ps(m14, e3);

        __m128 a0 = _mm_add_ps(m0, m1);
        __m128 a1 = _mm_add_ps(m2, m3);
        __m128 a2 = _mm_add_ps(a0, a1);

        _mm_storeu_ps(OutM.M[3], a2);
    }
}

void VSFastCrossSIMD(const Vector3 &InV1, const Vector3 &InV2, Vector3 &OutV)
{
    __m128 v1 = _mm_setr_ps(InV1.x, InV1.y, InV1.z, 0);
    __m128 v2 = _mm_setr_ps(InV2.x, InV2.y, InV2.z, 0);
    __m128 c = _mm_sub_ps(
        _mm_mul_ps(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(3, 1, 0, 2))),
        _mm_mul_ps(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(3, 0, 2, 1)))
    );
    OutV.x = *(float*)(&c);
    OutV.y = *((float*)(&c) + 1);
    OutV.z = *((float*)(&c) + 2);
}

void VSFastNormalizeSIMD(const Vector2 &InV, Vector2 &OutV)
{
    const float fThree = 3.0f;
    const float fOneHalf = 0.5f;
    __m128 ft = _mm_set_ss(fThree);
    __m128 fo = _mm_set_ss(fOneHalf);
    __m128 xy = _mm_setr_ps(InV.x, InV.y, 0, 0);
    __m128 xy2_ = _mm_mul_ps(xy, xy);
    __m128 yx2 = _mm_shuffle_ps(xy2_, xy2_, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 xy2 = _mm_shuffle_ps(xy2_, xy2_, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 x2y2 = _mm_add_ss(yx2, xy2);
    __m128 r_x2y2 = _mm_rsqrt_ss(x2y2);

    // 牛顿迭代
    __m128 f2 = _mm_sub_ss(ft, _mm_mul_ss(_mm_mul_ss(r_x2y2, x2y2), r_x2y2));
    __m128 f3 = _mm_and_ps(_mm_mul_ss(f2, _mm_mul_ss(r_x2y2, fo)),
        _mm_cmp_ss(_mm_xor_ps(f2, f2), x2y2, 4));
    f3 = _mm_shuffle_ps(f3, f3, 0);
    f3 = _mm_mul_ps(f3, xy);

    //__m128 v7 = _mm_mul_ps(xy, r_x2y2);
    OutV.x = *(float*)&f3;
    OutV.y = *((float*)&f3 + 1);
}

void VSFastNormalizeSIMD(const Vector3 &InV, Vector3 &OutV)
{
    const float fThree = 3.0f;
    const float fOneHalf = 0.5f;
    __m128 xyz = _mm_setr_ps(InV.x, InV.y, InV.z, 0);
    __m128 xyz2 = _mm_mul_ps(xyz, xyz);
    __m128 yyy2 = _mm_shuffle_ps(xyz2, xyz2, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 zzz2 = _mm_shuffle_ps(xyz2, xyz2, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 tmp = _mm_add_ps(_mm_add_ps(yyy2, zzz2), xyz2);
    __m128 x2y2z2 = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 r_x2y2z2 = _mm_rsqrt_ps(x2y2z2);
    // 牛顿迭代
    __m128 ft = _mm_set_ps1(fThree);
    __m128 fo = _mm_set_ps1(fOneHalf);
    __m128 a = _mm_and_ps(
        _mm_mul_ss(
            _mm_sub_ss(ft, 
                _mm_mul_ss(
                    _mm_mul_ss(r_x2y2z2, tmp), 
                    r_x2y2z2)),
        _mm_mul_ss(fo, r_x2y2z2)),
        _mm_cmp_ss(_mm_set_ps1(0.f), tmp, 4));// 精度比较
    __m128 a2 = _mm_shuffle_ps(a, a, 0);
    __m128 a3 = _mm_mul_ps(a2, xyz);

    //__m128 v8 = _mm_mul_ps(xyz, r_x2y2z2); // 最简单粗暴
    OutV.x = *(float*)&a3;
    OutV.y = *((float*)&a3 + 1);
    OutV.z = *((float*)&a3 + 2);
}

void VSFastNormalizeSIMD(const Vector3W &InV, Vector3W &OutV)
{
    const float fThree = 3.0f;
    const float fOneHalf = 0.5f;
    __m128 ft = _mm_set_ps1(fThree);
    __m128 fo = _mm_set_ps1(fOneHalf);

    __m128 xyzw = _mm_loadu_ps((const float*)&InV);
    __m128 xyzw2 = _mm_mul_ps(xyzw, xyzw);
    __m128 x2 = _mm_shuffle_ps(xyzw2, xyzw2, _MM_SHUFFLE(0, 0, 2, 0));
    __m128 y2 = _mm_shuffle_ps(xyzw2, xyzw2, _MM_SHUFFLE(0, 0, 3, 1));
    __m128 x2y2 = _mm_add_ps(x2, y2);
    __m128 z2 = _mm_shuffle_ps(x2y2, x2y2, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 x2y2z2 = _mm_add_ss(x2y2, z2);
    __m128 r = _mm_rsqrt_ss(x2y2z2);
    // 迭代
    __m128 f2 = _mm_mul_ss(
        _mm_sub_ss(ft, 
            _mm_mul_ss(
                _mm_mul_ss(r, x2y2z2), r)), 
        _mm_mul_ss(r, fo));

    __m128 r2 = _mm_and_ps(_mm_cmp_ss(_mm_xor_ps(r, r), x2y2z2, 4), f2);
    __m128 rt = _mm_mul_ps(xyzw, _mm_shuffle_ps(r2, r2, 0));

    _mm_storeu_ps(OutV.m, rt);
}

float VSFastLengthSIMD(const Vector3 &vec)
{
    const float fThree = 3.0f;
    const float fOneHalf = 0.5f;
    __m128 ft = _mm_set_ps1(fThree);
    __m128 fo = _mm_set_ps1(fOneHalf);

    __m128 xyz = _mm_setr_ps(vec.x, vec.y, vec.z, 0);
    __m128 xyz2 = _mm_mul_ps(xyz, xyz);

    __m128 y2 = _mm_shuffle_ps(xyz2, xyz2, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 z2 = _mm_shuffle_ps(xyz2, xyz2, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 x2y2z2 = _mm_add_ps(_mm_add_ps(y2, z2), xyz2);
    __m128 s_x2y2z2 = _mm_rsqrt_ps(x2y2z2);

    __m128 f1 = _mm_mul_ss(
        _mm_sub_ss(ft, 
            _mm_mul_ss(s_x2y2z2, 
                _mm_mul_ss(x2y2z2, s_x2y2z2))), 
        _mm_mul_ss(fo, s_x2y2z2));
    
    __m128 rt = _mm_and_ps(
        _mm_cmp_ss(
            _mm_xor_ps(ft, ft), x2y2z2, 4),
        _mm_mul_ss(f1, x2y2z2));

    return *(float*)&rt;
}

float VSFastLengthSIMD(const Vector3W &vec)
{
    const float fThree = 3.0f;
    const float fOneHalf = 0.5f;
    __m128 ft = _mm_set_ps1(fThree);
    __m128 fo = _mm_set_ps1(fOneHalf);

    __m128 xyzw = _mm_loadu_ps((const float*)&vec);
    __m128 xyzw2 = _mm_mul_ps(xyzw, xyzw);
    __m128 x2 = _mm_shuffle_ps(xyzw2, xyzw2, _MM_SHUFFLE(0, 0, 2, 0));
    __m128 y2 = _mm_shuffle_ps(xyzw2, xyzw2, _MM_SHUFFLE(0, 0, 3, 1));
    __m128 x2y2 = _mm_add_ps(x2, y2);
    __m128 z2 = _mm_shuffle_ps(x2y2, x2y2, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 x2y2z2 = _mm_add_ss(x2y2, z2);
    __m128 r = _mm_rsqrt_ss(x2y2z2);

    // 迭代
    __m128 f2 = _mm_mul_ss(
        _mm_sub_ss(ft,
            _mm_mul_ss(
                _mm_mul_ss(r, x2y2z2), r)),
        _mm_mul_ss(r, fo));
    __m128 rt = _mm_and_ps(
        _mm_mul_ss(f2, x2y2z2),
        _mm_cmp_ss(
            _mm_xor_ps(r, r), x2y2z2, 4));
    return *(float*)&rt;
}