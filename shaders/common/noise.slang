// Procedural noise of ui/renderer.py (hash2 / value_noise / fbm / smoothstep), evaluated per pixel.
module noise;

public float hash2(float x, float y)
{
    return frac(sin(x * 127.1f + y * 311.7f) * 43758.5453f);
}

public float value_noise(float x, float y)
{
    float xi = floor(x), yi = floor(y);
    float fx = x - xi, fy = y - yi;
    fx = fx * fx * (3.0f - 2.0f * fx);
    fy = fy * fy * (3.0f - 2.0f * fy);
    float a = hash2(xi, yi);
    float b = hash2(xi + 1.0f, yi);
    float c = hash2(xi, yi + 1.0f);
    float d = hash2(xi + 1.0f, yi + 1.0f);
    return lerp(lerp(a, b, fx), lerp(c, d, fx), fy);
}

public float fbm(float x, float y)
{
    return 0.5f * value_noise(x, y) + 0.3f * value_noise(2.1f * x, 2.1f * y) + 0.2f * value_noise(4.3f * x, 4.3f * y);
}

// Also valid with a > b (reversed ramp), unlike the GLSL builtin.
public float sstep(float a, float b, float x)
{
    float t = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
