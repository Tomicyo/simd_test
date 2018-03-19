#include "stdafx.h"
#include <xmmintrin.h>

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
    __m128 v1 = _mm_setr_ps(InV.x, InV.y, 0, 0);
    __m128 v2 = _mm_mul_ps(v1, v1);
    __m128 v3 = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(3, 2, 0, 1));
    __m128 v4 = _mm_shuffle_ps(v3, v3, _MM_SHUFFLE(3, 2, 0, 1));
    __m128 v5 = _mm_add_ps(v3, v4);
    __m128 v6 = _mm_rsqrt_ps(v5);
    __m128 v7 = _mm_mul_ps(v1, v6);
    OutV.x = *(float*)&v7;
    OutV.y = *((float*)&v7 + 1);
}

void VSFastNormalizeSIMD(const Vector3 &InV, Vector3 &OutV)
{
    const float fThree = 3.0f;
    const float fOneHalf = 0.5f;
    __m128 v1 = _mm_setr_ps(InV.x, InV.y, InV.z, 0);
    __m128 v2 = _mm_mul_ps(v1, v1);
    __m128 v3 = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 v4 = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 v5 = _mm_add_ps(_mm_add_ps(v3, v4), v2);
    __m128 v6 = _mm_shuffle_ps(v5, v5, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 v7 = _mm_rsqrt_ps(v6);
    __m128 v8 = _mm_mul_ps(v1, v7);
    OutV.x = *(float*)&v8;
    OutV.y = *((float*)&v8 + 1);
    OutV.z = *((float*)&v8 + 2);
}

void VSFastNormalizeSIMD(const Vector3W &InV, Vector3W &OutV)
{

}

float VSFastLengthSIMD(const Vector3 &vec)
{
    __m128 v1 = _mm_setr_ps(vec.x, vec.y, vec.z, 0);
    __m128 v2 = _mm_mul_ps(v1, v1);
    __m128 v3 = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 v4 = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 v5 = _mm_add_ps(_mm_add_ps(v3, v4), v2);
    __m128 v7 = _mm_sqrt_ps(v5);
    return *(float*)&v7;
}