// Palette and lighting of ui/renderer.py (display values, written to a UNORM framebuffer).
module palette;

public static const float3 SKY      = float3(0.62f, 0.78f, 0.92f);
public static const float3 GRASS_A  = float3(0.30f, 0.48f, 0.16f);
public static const float3 GRASS_B  = float3(0.42f, 0.58f, 0.20f);
public static const float3 SAND     = float3(0.86f, 0.76f, 0.56f);
public static const float3 WET_SAND = float3(0.60f, 0.52f, 0.36f);
public static const float3 SOIL     = float3(0.42f, 0.30f, 0.19f);
public static const float3 STONE    = float3(0.58f, 0.55f, 0.50f);
public static const float3 SHALLOW  = float3(0.16f, 0.58f, 0.62f);
public static const float3 DEEP     = float3(0.02f, 0.18f, 0.34f);
public static const float3 FOAM     = float3(0.95f, 0.97f, 1.0f);
public static const float3 MUD      = float3(0.42f, 0.31f, 0.18f);   // loaded flood water
public static const float3 MUD_DRY  = float3(0.55f, 0.47f, 0.36f);   // silt left by the flood
public static const float3 FOREST   = float3(0.13f, 0.24f, 0.10f);
public static const float3 MEADOW   = float3(0.40f, 0.56f, 0.22f);
public static const float3 CROP_A   = float3(0.66f, 0.60f, 0.34f);   // cereals
public static const float3 CROP_B   = float3(0.50f, 0.40f, 0.27f);   // ploughed

// Sky light (stronger upwards, reduced in hollows: occl) + Lambertian sun (masked in cast shadow: sun_vis)
public float3 light(float3 base, float3 n, float3 sun, float ambient, float diffuse, float occl, float sun_vis)
{
    float sky = ambient * (0.75f + 0.25f * n.y) * occl;
    return base * (sky + diffuse * max(dot(n, sun), 0.0f) * sun_vis);
}
