#include "geo/Http.h"

#include "util/Log.h"

#include <curl/curl.h>

#include <fstream>
#include <mutex>

namespace swe::geo::http {

namespace {

std::filesystem::path g_cacheDir = "cache/http";
std::mutex            g_cacheMutex;

constexpr char const *kUserAgent = "SWE-Vulkan/0.2 (shallow-water flood simulator; https://github.com/)";

void ensureCurl()
{
	static std::once_flag once;
	std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

size_t writeBody(char *ptr, size_t size, size_t nmemb, void *user)
{
	auto *body = static_cast<std::vector<uint8_t> *>(user);
	body->insert(body->end(), reinterpret_cast<uint8_t *>(ptr), reinterpret_cast<uint8_t *>(ptr) + size * nmemb);
	return size * nmemb;
}

// FNV-1a 64 of the URL: cache file name
std::string hashUrl(std::string const &url)
{
	uint64_t h = 1469598103934665603ull;
	for (unsigned char c : url)
	{
		h ^= c;
		h *= 1099511628211ull;
	}
	char buf[17];
	std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
	return buf;
}

}        // namespace

void setCacheDir(std::filesystem::path const &dir)
{
	std::lock_guard lock(g_cacheMutex);
	g_cacheDir = dir;
}

std::filesystem::path cacheDir()
{
	std::lock_guard lock(g_cacheMutex);
	return g_cacheDir;
}

std::filesystem::path cachePath(std::string const &url)
{
	auto h = hashUrl(url);
	return cacheDir() / h.substr(0, 2) / (h + ".bin");
}

std::string urlEncode(std::string const &s)
{
	ensureCurl();
	char *e = curl_easy_escape(nullptr, s.c_str(), static_cast<int>(s.size()));
	std::string out = e ? e : s;
	curl_free(e);
	return out;
}

Response get(std::string const &url, bool cache, int timeoutSeconds)
{
	Response r;
	auto     path = cachePath(url);
	if (cache)
	{
		std::ifstream f(path, std::ios::binary);
		if (f)
		{
			std::ifstream meta(path.string() + ".type");
			std::getline(meta, r.contentType);
			r.body.assign(std::istreambuf_iterator<char>(f), {});
			r.status    = 200;
			r.fromCache = true;
			return r;
		}
	}

	ensureCurl();
	CURL *curl = curl_easy_init();
	if (!curl)
	{
		r.error = "curl_easy_init failed";
		return r;
	}
	curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
	curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
	curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
	curl_easy_setopt(curl, CURLOPT_USERAGENT, kUserAgent);
	curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, static_cast<long>(timeoutSeconds));
	curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 20L);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &writeBody);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &r.body);
	curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
	CURLcode code = curl_easy_perform(curl);
	if (code != CURLE_OK)
		r.error = curl_easy_strerror(code);
	else
	{
		curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &r.status);
		char *ct = nullptr;
		curl_easy_getinfo(curl, CURLINFO_CONTENT_TYPE, &ct);
		if (ct)
			r.contentType = ct;
	}
	curl_easy_cleanup(curl);

	if (r.ok() && cache && !r.body.empty())
	{
		std::error_code ec;
		std::filesystem::create_directories(path.parent_path(), ec);
		std::ofstream out(path, std::ios::binary);
		out.write(reinterpret_cast<char const *>(r.body.data()), static_cast<std::streamsize>(r.body.size()));
		std::ofstream(path.string() + ".type") << r.contentType;
	}
	else if (!r.ok())
		log::warn("HTTP {} {} : {}", r.status, url.substr(0, 120), r.error);
	return r;
}

}        // namespace swe::geo::http
