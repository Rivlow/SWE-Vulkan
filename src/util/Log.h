#pragma once
#include <format>
#include <iostream>
#include <string_view>

namespace swe::log {

template <class... Args>
void info(std::format_string<Args...> fmt, Args &&...args)
{
	std::cout << "[swe] " << std::format(fmt, std::forward<Args>(args)...) << std::endl;        // flushed: logs survive a kill
}

template <class... Args>
void warn(std::format_string<Args...> fmt, Args &&...args)
{
	std::cerr << "[swe] warning: " << std::format(fmt, std::forward<Args>(args)...) << '\n';
}

template <class... Args>
void error(std::format_string<Args...> fmt, Args &&...args)
{
	std::cerr << "[swe] error: " << std::format(fmt, std::forward<Args>(args)...) << '\n';
}

}        // namespace swe::log
