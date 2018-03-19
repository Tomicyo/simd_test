// Main.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <assert.h>

int main()
{
    Matrix3x3W a = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 2.0f, -3.0f, 4.0f,
        1.0f, -2.0f, 3.0f, 4.0f,
    };
    Matrix3x3W b = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 2.0f, -3.0f, 4.0f,
        1.0f, -2.0f, 3.0f, 4.0f,
    };
    Matrix3x3W c;
    Matrix3x3W d;
    VSFastMulASM(a, b, c);
    VSFastMulSIMD(a, b, d);
    assert(Equal(c, d));

    Vector3 v1 = { 0.2f, 1.0f, 0.6f };
    Vector3 v2 = { 0.2f, -1.0f, 0.1f };
    Vector3 v3; // 0.7, 0.1, -0.4
    Vector3 v4;
    VSFastCrossASM(v1, v2, v3);
    VSFastCrossSIMD(v1, v2, v4);
    assert(Equal(v3, v4));

    Vector2 n1 = {0.3, 0.4};
    Vector2 n2,n3;
    VSFastNormalizeASM(n1, n2);
    VSFastNormalizeSIMD(n1, n3);
    assert(Equal(n2, n3));


    Vector3 n31 = { 0.3, 0.2, 0.1 };
    Vector3 n32, n33;
    VSFastNormalizeASM(n31, n32);
    VSFastNormalizeSIMD(n31, n33);
    assert(Equal(n32, n33));

    float r = VSFastLengthSIMD(v3);
    float r1 = VSFastLengthASM(v3);
    assert(Equal(r, r1));
    return 0;
}

