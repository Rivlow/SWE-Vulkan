#include "util/Png.h"

#include "util/Log.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <vector>

namespace swe {

bool writePng(std::filesystem::path const &path, uint8_t const *pixels, uint32_t width, uint32_t height, bool bgra)
{
	if (path.has_parent_path())
		std::filesystem::create_directories(path.parent_path());
	std::vector<uint8_t> rgba(size_t(width) * height * 4);
	for (size_t k = 0; k < size_t(width) * height; ++k)
	{
		rgba[4 * k + 0] = pixels[4 * k + (bgra ? 2 : 0)];
		rgba[4 * k + 1] = pixels[4 * k + 1];
		rgba[4 * k + 2] = pixels[4 * k + (bgra ? 0 : 2)];
		rgba[4 * k + 3] = 255;
	}
	if (!stbi_write_png(path.string().c_str(), int(width), int(height), 4, rgba.data(), int(width) * 4))
	{
		log::error("cannot write {}", path.string());
		return false;
	}
	return true;
}

}        // namespace swe
