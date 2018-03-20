#include "stdafx.h"
#include <xmmintrin.h>

void VSFastMulASM(const Matrix3x3W & InM1, const Matrix3x3W & InM2, Matrix3x3W & OutM)
{
    __asm
    {
        mov		eax, [InM2]
        mov		ecx, [InM1]
        movups	xmm4, [eax]
        movups	xmm5, [eax + 16]
        movups	xmm6, [eax + 32]
        movups	xmm7, [eax + 48] // _mm_loadu_ps 

        // 1
        movss	xmm0, [ecx] //_mm_load_ss
        shufps	xmm0, xmm0, 0 //_mm_shuffle_ps 
        mulps	xmm0, xmm4

        movss	xmm1, [ecx + 4]
        shufps	xmm1, xmm1, 0
        mulps	xmm1, xmm5

        movss	xmm2, [ecx + 8]
        shufps	xmm2, xmm2, 0
        mulps	xmm2, xmm6

        addps	xmm1, xmm0			// Row 1

        movss	xmm3, [ecx + 12]
        shufps	xmm3, xmm3, 0
        mulps	xmm3, xmm7

        // 2
        movss	xmm0, [ecx + 16]
        shufps	xmm0, xmm0, 0
        mulps	xmm0, xmm4

        addps	xmm3, xmm2			// Row 1

        movss	xmm2, [ecx + 20]
        shufps	xmm2, xmm2, 0
        mulps	xmm2, xmm5

        addps	xmm3, xmm1			// Row 1

        mov		eax, [OutM]

        movss	xmm1, [ecx + 24]
        shufps	xmm1, xmm1, 0
        mulps	xmm1, xmm6

        movups[eax], xmm3			// Row 1 out

        addps	xmm2, xmm0			// Row 2

        movss	xmm3, [ecx + 28]
        shufps	xmm3, xmm3, 0
        mulps	xmm3, xmm7

        // 3
        movss	xmm0, [ecx + 32]
        shufps	xmm0, xmm0, 0
        mulps	xmm0, xmm4

        addps	xmm3, xmm1			// Row 2

        movss	xmm1, [ecx + 36]
        shufps	xmm1, xmm1, 0
        mulps	xmm1, xmm5

        addps	xmm3, xmm2			// Row 2

        movss	xmm2, [ecx + 40]
        shufps	xmm2, xmm2, 0
        mulps	xmm2, xmm6

        movups[eax + 16], xmm3		// Row 2 out

        addps	xmm1, xmm0			// Row 3

        movss	xmm3, [ecx + 44]
        shufps	xmm3, xmm3, 0
        mulps	xmm3, xmm7

        // 4
        movss	xmm0, [ecx + 48]
        shufps	xmm0, xmm0, 0
        mulps	xmm0, xmm4

        addps	xmm3, xmm2			// Row 3

        movss	xmm2, [ecx + 52]
        shufps	xmm2, xmm2, 0
        mulps	xmm2, xmm5

        addps	xmm3, xmm1			// Row 3

        movss	xmm1, [ecx + 56]
        shufps	xmm1, xmm1, 0
        mulps	xmm1, xmm6

        movups[eax + 32], xmm3		// Row 3 out

        addps	xmm2, xmm0			// Row 4

        movss	xmm3, [ecx + 60]
        shufps	xmm3, xmm3, 0
        mulps	xmm3, xmm7

        addps	xmm3, xmm1			// Row 4

        addps	xmm3, xmm2			// Row 4

        movups[eax + 48], xmm3		// Row 4 out
    }
}

void VSFastCrossASM(const Vector3 &InV1, const Vector3 &InV2, Vector3 &OutV)
{
    __asm
    {
        mov ecx, [InV1];
        mov eax, [InV2];

        movups xmm0, [ecx];
        movups xmm1, [eax];

        movaps xmm2, xmm0;
        movaps xmm3, xmm1;

        shufps xmm0, xmm0, _MM_SHUFFLE(0, 0, 2, 1);
        shufps xmm1, xmm1, _MM_SHUFFLE(0, 1, 0, 2);

        mulps xmm0, xmm1;

        mov eax, [OutV];

        shufps xmm2, xmm2, _MM_SHUFFLE(0, 1, 0, 2);
        shufps xmm3, xmm3, _MM_SHUFFLE(0, 0, 2, 1);

        mulps xmm2, xmm3;
        subps xmm0, xmm2;

        movss[eax], xmm0;
        shufps xmm0, xmm0, _MM_SHUFFLE(3, 2, 1, 1);
        movss[eax + 4], xmm0;
        shufps xmm0, xmm0, _MM_SHUFFLE(3, 2, 1, 2);
        movss[eax + 8], xmm0;
    }
}

void VSFastNormalizeASM(const Vector2 &InV, Vector2 &OutV)
{
    const float fThree = 3.0f;
    const float fOneHalf = 0.5f;
    __asm
    {
        mov ecx, [InV];
        movups xmm0, [ecx];
        movaps xmm5, xmm0;
        mulps xmm0, xmm0;

        movaps xmm1, xmm0;
        shufps xmm0, xmm0, _MM_SHUFFLE(0, 0, 0, 0);
        shufps xmm1, xmm1, _MM_SHUFFLE(1, 1, 1, 1);

        addss xmm1, xmm0;
        rsqrtss xmm0, xmm1;
        // Newton-Raphson iteration
        movss xmm3, [fThree];
        movss xmm2, xmm0;
        mulss xmm0, xmm1;
        mulss xmm0, xmm2;
        subss xmm3, xmm0;
        mulss xmm2, [fOneHalf];
        mulss xmm3, xmm2;

        xorps xmm4, xmm4;
        cmpss xmm4, xmm1, 4;

        andps xmm3, xmm4;

        shufps xmm3, xmm3, 0;
        mulps xmm3, xmm5;
        mov ecx, [OutV];

        movss[ecx], xmm3;

        shufps xmm3, xmm3, _MM_SHUFFLE(1, 1, 1, 1);
        movss[ecx + 4], xmm3;
    }
}

void VSFastNormalizeASM(const Vector3 &InV, Vector3 &OutV)
{
    const float fThree = 3.0f;
    const float fOneHalf = 0.5f;
    __asm
    {
        mov ecx, [InV];
        movups xmm0, [ecx];
        movaps xmm5, xmm0;
        mulps xmm0, xmm0;

        movaps xmm1, xmm0;
        movaps xmm2, xmm0;
        shufps xmm1, xmm1, _MM_SHUFFLE(1, 1, 1, 1);
        shufps xmm2, xmm2, _MM_SHUFFLE(2, 2, 2, 2);

        addss xmm1, xmm0;
        addss xmm1, xmm2;
        rsqrtss xmm0, xmm1;
        // Newton-Raphson iteration
        movss xmm3, [fThree];
        movss xmm2, xmm0;
        mulss xmm0, xmm1;
        mulss xmm0, xmm2;
        mulss xmm2, [fOneHalf];
        subss xmm3, xmm0;
        mulss xmm3, xmm2;

        xorps xmm4, xmm4; // 0
        cmpss xmm4, xmm1, 4; // avx

        andps xmm3, xmm4;

        shufps xmm3, xmm3, 0;
        mulps xmm3, xmm5;
        mov ecx, [OutV];

        movss[ecx], xmm3;

        shufps xmm3, xmm3, _MM_SHUFFLE(3, 2, 1, 1);
        movss[ecx + 4], xmm3;

        shufps xmm3, xmm3, _MM_SHUFFLE(3, 2, 1, 2);
        movss[ecx + 8], xmm3;
    }
}

void VSFastNormalizeASM(const Vector3W &InV, Vector3W &OutV)
{
    const float fThree = 3.0f;
    const float fOneHalf = 0.5f;
    __asm
    {
        mov ecx, [InV];
        movups xmm0, [ecx];
        movaps xmm5, xmm0;
        mulps xmm0, xmm0;

        movaps xmm1, xmm0;

        shufps xmm0, xmm0, _MM_SHUFFLE(0, 0, 2, 0);
        shufps xmm1, xmm1, _MM_SHUFFLE(0, 0, 3, 1);

        addps xmm1, xmm0;

        movaps xmm2, xmm1;
        shufps xmm2, xmm2, _MM_SHUFFLE(1, 1, 1, 1);

        addss xmm1, xmm2;

        rsqrtss xmm0, xmm1;
        // Newton-Raphson iteration
        movss xmm3, [fThree];
        movss xmm2, xmm0;
        mulss xmm0, xmm1;
        mulss xmm0, xmm2;
        mulss xmm2, [fOneHalf];
        subss xmm3, xmm0;
        mulss xmm3, xmm2;

        xorps xmm4, xmm4;
        cmpss xmm4, xmm1, 4;

        andps xmm3, xmm4;

        shufps xmm3, xmm3, 0;
        mulps xmm3, xmm5;
        mov ecx, [OutV];

        movups[ecx], xmm3;
    }
}

float VSFastLengthASM(const Vector3 &vec)
{
    float rt = 0.f;
    const float fThree = 3.0f;
    const float fOneHalf = 0.5f;
    __asm
    {
        mov ecx, [vec];
        movups xmm0, [ecx];
        mulps xmm0, xmm0;

        movaps xmm1, xmm0;
        movaps xmm2, xmm0;
        shufps xmm1, xmm1, _MM_SHUFFLE(1, 1, 1, 1);
        shufps xmm2, xmm2, _MM_SHUFFLE(2, 2, 2, 2);

        addss xmm1, xmm0;
        addss xmm1, xmm2;
        rsqrtss xmm0, xmm1;
        // Newton-Raphson iteration
        movss xmm3, [fThree];
        movss xmm2, xmm0;
        mulss xmm0, xmm1;
        mulss xmm0, xmm2;
        mulss xmm2, [fOneHalf];
        subss xmm3, xmm0;
        mulss xmm3, xmm2;

        xorps xmm4, xmm4;
        cmpss xmm4, xmm1, 4;

        mulss xmm3, xmm1;
        andps xmm3, xmm4;
        movss[rt], xmm3;
    }
    return rt;
}

float VSFastLengthASM(const Vector3W &vec)
{
    float rt = 0.f;
    const float fThree = 3.0f;
    const float fOneHalf = 0.5f;
    __asm
    {
        mov ecx, [vec];
        movups xmm0, [ecx];
        mulps xmm0, xmm0;

        movaps xmm1, xmm0;
        shufps xmm0, xmm0, _MM_SHUFFLE(0, 0, 2, 0);
        shufps xmm1, xmm1, _MM_SHUFFLE(0, 0, 3, 1);

        addps xmm1, xmm0;

        movaps xmm2, xmm1;
        shufps xmm2, xmm2, _MM_SHUFFLE(1, 1, 1, 1);

        addss xmm1, xmm2;

        rsqrtss xmm0, xmm1;
        // Newton-Raphson iteration
        movss xmm3, [fThree];
        movss xmm2, xmm0;
        mulss xmm0, xmm1;
        mulss xmm0, xmm2;
        mulss xmm2, [fOneHalf];
        subss xmm3, xmm0;
        mulss xmm3, xmm2;

        xorps xmm4, xmm4;
        cmpss xmm4, xmm1, 4;

        mulss xmm3, xmm1;
        andps xmm3, xmm4;
        movss[rt], xmm3;
    }
    return rt;
}