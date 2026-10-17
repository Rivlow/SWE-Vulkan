#include "cases/Npy.h"

#include <cstring>
#include <fstream>
#include <regex>
#include <stdexcept>

namespace swe::npy {

namespace {

std::vector<uint8_t> readFile(std::filesystem::path const &path)
{
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f)
		throw std::runtime_error("cannot open " + path.string());
	std::vector<uint8_t> bytes(static_cast<size_t>(f.tellg()));
	f.seekg(0);
	f.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
	return bytes;
}

template <class T>
T readLE(uint8_t const *p)
{
	T v;
	std::memcpy(&v, p, sizeof(T));        // x64 is little-endian
	return v;
}

template <class Out>
std::vector<Out> convert(Array const &a)
{
	size_t           n = a.size();
	std::vector<Out> out(n);
	auto const      *p = a.data.data();
	auto const      &d = a.descr;
	for (size_t k = 0; k < n; ++k)
	{
		if (d == "<f4")
			out[k] = static_cast<Out>(readLE<float>(p + 4 * k));
		else if (d == "<f8")
			out[k] = static_cast<Out>(readLE<double>(p + 8 * k));
		else if (d == "<i2")
			out[k] = static_cast<Out>(readLE<int16_t>(p + 2 * k));
		else if (d == "<i4")
			out[k] = static_cast<Out>(readLE<int32_t>(p + 4 * k));
		else if (d == "<i8")
			out[k] = static_cast<Out>(readLE<int64_t>(p + 8 * k));
		else if (d == "|u1" || d == "|b1" || d == "|i1")
			out[k] = static_cast<Out>(p[k]);
		else
			throw std::runtime_error("npy: unsupported dtype " + d);
	}
	return out;
}

}        // namespace

size_t Array::itemSize() const
{
	if (descr.size() < 3)
		throw std::runtime_error("npy: bad descr " + descr);
	return static_cast<size_t>(std::stoi(descr.substr(2)));
}

size_t Array::size() const
{
	size_t n = 1;
	for (auto s : shape)
		n *= s;
	return n;
}

std::vector<float>  Array::asFloat() const { return convert<float>(*this); }
std::vector<double> Array::asDouble() const { return convert<double>(*this); }

Array parse(uint8_t const *bytes, size_t length, std::string const &what)
{
	static constexpr char magic[] = "\x93NUMPY";
	if (length < 10 || std::memcmp(bytes, magic, 6) != 0)
		throw std::runtime_error("npy: not a .npy file: " + what);
	uint8_t major     = bytes[6];
	size_t  headerLen = 0, offset = 0;
	if (major == 1)
	{
		headerLen = readLE<uint16_t>(bytes + 8);
		offset    = 10;
	}
	else
	{
		headerLen = readLE<uint32_t>(bytes + 8);
		offset    = 12;
	}
	std::string header(reinterpret_cast<char const *>(bytes + offset), headerLen);

	std::smatch m;
	Array       a;
	if (!std::regex_search(header, m, std::regex(R"('descr'\s*:\s*'([^']+)')")))
		throw std::runtime_error("npy: no descr in " + what);
	a.descr = m[1];
	if (std::regex_search(header, m, std::regex(R"('fortran_order'\s*:\s*True)")))
		throw std::runtime_error("npy: fortran_order arrays are not supported: " + what);
	if (!std::regex_search(header, m, std::regex(R"('shape'\s*:\s*\(([^)]*)\))")))
		throw std::runtime_error("npy: no shape in " + what);
	std::string shape = m[1];
	std::regex  num(R"(\d+)");
	for (auto it = std::sregex_iterator(shape.begin(), shape.end(), num); it != std::sregex_iterator(); ++it)
		a.shape.push_back(std::stoull(it->str()));

	size_t dataOffset = offset + headerLen;
	size_t bytesNeeded = a.size() * a.itemSize();
	if (dataOffset + bytesNeeded > length)
		throw std::runtime_error("npy: truncated data in " + what);
	a.data.assign(bytes + dataOffset, bytes + dataOffset + bytesNeeded);
	return a;
}

Array load(std::filesystem::path const &path)
{
	auto bytes = readFile(path);
	return parse(bytes.data(), bytes.size(), path.string());
}

namespace {

// Complete .npy file image (header + data)
std::vector<uint8_t> encodeNpy(std::string const &descr, std::vector<size_t> const &shape, void const *data, size_t bytes)
{
	std::string shapeStr = "(";
	for (size_t k = 0; k < shape.size(); ++k)
		shapeStr += std::to_string(shape[k]) + (shape.size() == 1 ? "," : (k + 1 < shape.size() ? ", " : ""));
	shapeStr += ")";
	std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shapeStr + ", }";
	// magic(6) + version(2) + len(2) + header + '\n' padded to a multiple of 64
	size_t total = 10 + header.size() + 1;
	header.append((64 - total % 64) % 64, ' ');
	header += '\n';

	std::vector<uint8_t> out;
	out.reserve(10 + header.size() + bytes);
	char const magic[] = "\x93NUMPY\x01\x00";
	out.insert(out.end(), magic, magic + 8);
	auto len = static_cast<uint16_t>(header.size());
	out.push_back(static_cast<uint8_t>(len & 0xff));
	out.push_back(static_cast<uint8_t>(len >> 8));
	out.insert(out.end(), header.begin(), header.end());
	auto const *p = static_cast<uint8_t const *>(data);
	out.insert(out.end(), p, p + bytes);
	return out;
}

uint32_t crc32(uint8_t const *data, size_t n)
{
	static uint32_t table[256];
	static bool     init = false;
	if (!init)
	{
		for (uint32_t i = 0; i < 256; ++i)
		{
			uint32_t c = i;
			for (int k = 0; k < 8; ++k)
				c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
			table[i] = c;
		}
		init = true;
	}
	uint32_t c = 0xFFFFFFFFu;
	for (size_t k = 0; k < n; ++k)
		c = table[(c ^ data[k]) & 0xff] ^ (c >> 8);
	return c ^ 0xFFFFFFFFu;
}

template <class T>
void putLE(std::vector<uint8_t> &out, T v)
{
	for (size_t k = 0; k < sizeof(T); ++k)
		out.push_back(static_cast<uint8_t>((static_cast<uint64_t>(v) >> (8 * k)) & 0xff));
}

}        // namespace

void save(std::filesystem::path const &path, std::string const &descr, std::vector<size_t> const &shape, void const *data,
          size_t bytes)
{
	auto          image = encodeNpy(descr, shape, data, bytes);
	std::ofstream f(path, std::ios::binary);
	if (!f)
		throw std::runtime_error("cannot write " + path.string());
	f.write(reinterpret_cast<char const *>(image.data()), static_cast<std::streamsize>(image.size()));
}

void saveNpz(std::filesystem::path const &path, std::map<std::string, Array> const &arrays)
{
	std::vector<uint8_t> zip, central;
	uint16_t             count = 0;
	for (auto const &[name, a] : arrays)
	{
		auto     image = encodeNpy(a.descr, a.shape, a.data.data(), a.data.size());
		auto     crc   = crc32(image.data(), image.size());
		auto     size  = static_cast<uint32_t>(image.size());
		auto     off   = static_cast<uint32_t>(zip.size());
		std::string fname = name + ".npy";
		// local file header (stored)
		putLE<uint32_t>(zip, 0x04034b50);
		putLE<uint16_t>(zip, 20);
		putLE<uint16_t>(zip, 0);
		putLE<uint16_t>(zip, 0);
		putLE<uint16_t>(zip, 0);
		putLE<uint16_t>(zip, 0);
		putLE<uint32_t>(zip, crc);
		putLE<uint32_t>(zip, size);
		putLE<uint32_t>(zip, size);
		putLE<uint16_t>(zip, static_cast<uint16_t>(fname.size()));
		putLE<uint16_t>(zip, 0);
		zip.insert(zip.end(), fname.begin(), fname.end());
		zip.insert(zip.end(), image.begin(), image.end());
		// central directory entry
		putLE<uint32_t>(central, 0x02014b50);
		putLE<uint16_t>(central, 20);
		putLE<uint16_t>(central, 20);
		putLE<uint16_t>(central, 0);
		putLE<uint16_t>(central, 0);
		putLE<uint16_t>(central, 0);
		putLE<uint16_t>(central, 0);
		putLE<uint32_t>(central, crc);
		putLE<uint32_t>(central, size);
		putLE<uint32_t>(central, size);
		putLE<uint16_t>(central, static_cast<uint16_t>(fname.size()));
		putLE<uint16_t>(central, 0);
		putLE<uint16_t>(central, 0);
		putLE<uint16_t>(central, 0);
		putLE<uint16_t>(central, 0);
		putLE<uint32_t>(central, 0);
		putLE<uint32_t>(central, off);
		central.insert(central.end(), fname.begin(), fname.end());
		++count;
	}
	auto cdOff = static_cast<uint32_t>(zip.size());
	zip.insert(zip.end(), central.begin(), central.end());
	putLE<uint32_t>(zip, 0x06054b50);
	putLE<uint16_t>(zip, 0);
	putLE<uint16_t>(zip, 0);
	putLE<uint16_t>(zip, count);
	putLE<uint16_t>(zip, count);
	putLE<uint32_t>(zip, static_cast<uint32_t>(central.size()));
	putLE<uint32_t>(zip, cdOff);
	putLE<uint16_t>(zip, 0);

	std::ofstream f(path, std::ios::binary);
	if (!f)
		throw std::runtime_error("cannot write " + path.string());
	f.write(reinterpret_cast<char const *>(zip.data()), static_cast<std::streamsize>(zip.size()));
}

Array makeArray(std::vector<float> const &v, std::vector<size_t> const &shape)
{
	Array a;
	a.descr = "<f4";
	a.shape = shape;
	a.data.assign(reinterpret_cast<uint8_t const *>(v.data()), reinterpret_cast<uint8_t const *>(v.data()) + v.size() * 4);
	return a;
}

Array makeArray(std::vector<uint8_t> const &v, std::vector<size_t> const &shape)
{
	Array a;
	a.descr = "|u1";
	a.shape = shape;
	a.data  = v;
	return a;
}

Array makeArray(std::vector<double> const &v, std::vector<size_t> const &shape)
{
	Array a;
	a.descr = "<f8";
	a.shape = shape;
	a.data.assign(reinterpret_cast<uint8_t const *>(v.data()), reinterpret_cast<uint8_t const *>(v.data()) + v.size() * 8);
	return a;
}

std::map<std::string, Array> loadNpz(std::filesystem::path const &path)
{
	auto        bytes = readFile(path);
	auto const *b     = bytes.data();
	size_t      n     = bytes.size();

	// End of central directory record (search backwards, a comment may follow it)
	size_t eocd = std::string::npos;
	if (n >= 22)
		for (size_t k = n - 22 + 1; k-- > 0;)
			if (readLE<uint32_t>(b + k) == 0x06054b50)
			{
				eocd = k;
				break;
			}
	if (eocd == std::string::npos)
		throw std::runtime_error("npz: no end-of-central-directory in " + path.string());

	uint16_t entries = readLE<uint16_t>(b + eocd + 10);
	uint32_t cdOff   = readLE<uint32_t>(b + eocd + 16);

	std::map<std::string, Array> out;
	size_t                       p = cdOff;
	for (uint16_t e = 0; e < entries; ++e)
	{
		if (readLE<uint32_t>(b + p) != 0x02014b50)
			throw std::runtime_error("npz: bad central directory in " + path.string());
		uint16_t    method   = readLE<uint16_t>(b + p + 10);
		uint32_t    compSize = readLE<uint32_t>(b + p + 20);
		uint16_t    nameLen  = readLE<uint16_t>(b + p + 28);
		uint16_t    extraLen = readLE<uint16_t>(b + p + 30);
		uint16_t    cmtLen   = readLE<uint16_t>(b + p + 32);
		uint32_t    localOff = readLE<uint32_t>(b + p + 42);
		std::string name(reinterpret_cast<char const *>(b + p + 46), nameLen);
		p += 46 + nameLen + extraLen + cmtLen;

		if (method != 0)
			throw std::runtime_error("npz: entry '" + name + "' is compressed (use np.savez, not savez_compressed): " + path.string());
		if (readLE<uint32_t>(b + localOff) != 0x04034b50)
			throw std::runtime_error("npz: bad local header in " + path.string());
		uint16_t lNameLen  = readLE<uint16_t>(b + localOff + 26);
		uint16_t lExtraLen = readLE<uint16_t>(b + localOff + 28);
		size_t   dataStart = localOff + 30 + lNameLen + lExtraLen;
		if (dataStart + compSize > n)
			throw std::runtime_error("npz: truncated entry '" + name + "'");

		if (name.size() > 4 && name.ends_with(".npy"))
			name.resize(name.size() - 4);
		out[name] = parse(b + dataStart, compSize, path.string() + ":" + name);
	}
	return out;
}

}        // namespace swe::npy
