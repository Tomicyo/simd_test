#include "stdafx.h"
#include <math.h>

bool Equal(Matrix3x3W const&a, Matrix3x3W const&b)
{
    for (int i = 0; i < 16; i++)
    {
        if (fabs(a.m[i] - b.m[i]) > 1e-6f)
        {
            return false;
        }
    }
    return true;
}

bool Equal(Vector2 const&a, Vector2 const&b)
{
    return fabs(a.x - b.x) < 1e-6f && fabs(a.y - b.y) < 1e-6f;
}

bool Equal(Vector3 const&a, Vector3 const&b)
{
    return fabs(a.x - b.x) < 1e-6f 
        && fabs(a.y - b.y) < 1e-6f
        && fabs(a.z - b.z) < 1e-6f;
}

bool Equal(float const&a, float const&b)
{
    return fabs(a - b) < 1e-6f;
}

bool Equal(Vector3W const&a, Vector3W const&b)
{
    return fabs(a.x - b.x) < 1e-6f
        && fabs(a.y - b.y) < 1e-6f
        && fabs(a.z - b.z) < 1e-6f
        && fabs(a.w - b.w) < 1e-6f;
}