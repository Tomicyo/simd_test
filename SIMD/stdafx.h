// stdafx.h : 标准系统包含文件的包含文件，
// 或是经常使用但不常更改的
// 特定于项目的包含文件
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>


typedef union
{
    float m[16];
    struct
    {
        float _00, _01, _02, _03;
        float _10, _11, _12, _13;
        float _20, _21, _22, _23;
        float _30, _31, _32, _33;
    };
    float M[4][4];
}Matrix3x3W;

typedef union
{
    float m[2];
    struct
    {
        float x, y;
    };
}Vector2;

typedef union
{
    float m[3];
    struct
    {
        float x, y, z;
    };
}Vector3;

typedef union
{
    float m[4];
    struct
    {
        float x, y, z, w;
    };
    struct
    {
        float r, g, b, a;
    };
}Vector3W;

extern bool Equal(float const&a, float const&b);
extern bool Equal(Matrix3x3W const&a, Matrix3x3W const&b);
extern bool Equal(Vector2 const&a, Vector2 const&b);
extern bool Equal(Vector3 const&a, Vector3 const&b);
extern bool Equal(Vector3W const&a, Vector3W const&b);

extern void VSFastMulASM(const Matrix3x3W & InM1, const Matrix3x3W & InM2, Matrix3x3W & OutM);
extern void VSFastMulSIMD(const Matrix3x3W & InM1, const Matrix3x3W & InM2, Matrix3x3W & OutM);

extern void VSFastCrossASM(const Vector3 &InV1, const Vector3 &InV2, Vector3 &OutV);
extern void VSFastCrossSIMD(const Vector3 &InV1, const Vector3 &InV2, Vector3 &OutV);

extern void VSFastNormalizeASM(const Vector2 &InV, Vector2 &OutV);
extern void VSFastNormalizeSIMD(const Vector2 &InV, Vector2 &OutV);

extern void VSFastNormalizeASM(const Vector3 &InV, Vector3 &OutV);
extern void VSFastNormalizeSIMD(const Vector3 &InV, Vector3 &OutV);

extern void VSFastNormalizeASM(const Vector3W &InV, Vector3W &OutV);
extern void VSFastNormalizeSIMD(const Vector3W &InV, Vector3W &OutV);

extern float VSFastLengthASM(const Vector3 &vec);
extern float VSFastLengthSIMD(const Vector3 &vec);

extern float VSFastLengthASM(const Vector3W &vec);
extern float VSFastLengthSIMD(const Vector3W &vec);