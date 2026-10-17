// Turbo colour map (Google, polynomial approximation, Apache-2.0). Same coefficients in src/render/Colormap.cpp.
module colormap;

public float3 turbo(float x)
{
    const float4 kRedVec4   = float4(0.13572138f, 4.61539260f, -42.66032258f, 132.13108234f);
    const float4 kGreenVec4 = float4(0.09140261f, 2.19418839f, 4.84296658f, -14.18503333f);
    const float4 kBlueVec4  = float4(0.10667330f, 12.64194608f, -60.58204836f, 110.36276771f);
    const float2 kRedVec2   = float2(-152.94239396f, 59.28637943f);
    const float2 kGreenVec2 = float2(4.27729857f, 2.82956604f);
    const float2 kBlueVec2  = float2(-89.90310912f, 27.34824973f);
    x = saturate(x);
    float4 v4 = float4(1.0f, x, x * x, x * x * x);
    float2 v2 = v4.zw * v4.z;
    return float3(dot(v4, kRedVec4) + dot(v2, kRedVec2),
                  dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
                  dot(v4, kBlueVec4) + dot(v2, kBlueVec2));
}
