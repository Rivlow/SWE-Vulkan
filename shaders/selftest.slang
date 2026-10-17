// M0 self-test: fills buf[i] = i * 2 + 1 so the host can check the sum.
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> buf;

struct PC { uint count; };
[[vk::push_constant]] ConstantBuffer<PC> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void fill(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= pc.count) return;
    buf[tid.x] = tid.x * 2 + 1;
}
