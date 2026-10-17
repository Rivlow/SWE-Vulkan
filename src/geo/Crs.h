#pragma once
// Coordinate systems used by the geographic import.
//   Lambert 72  (EPSG:31370) : the solver grid (metres, x east, y north) - Belgian official grid
//   WGS 84      (EPSG:4326)  : lon/lat for the user (Google Maps links, GPS)
//   Web Mercator(EPSG:3857)  : the slippy map tiles
// Lambert 72 <-> WGS 84 goes through PROJ (via GDAL's OGR), Web Mercator is closed-form.
#include <memory>
#include <string>

namespace swe::geo {

struct LonLat
{
	double lon = 0, lat = 0;        // degrees
};
struct XY
{
	double x = 0, y = 0;            // metres
};
struct BBox
{
	double xmin = 0, ymin = 0, xmax = 0, ymax = 0;
	double width() const { return xmax - xmin; }
	double height() const { return ymax - ymin; }
	bool   empty() const { return xmax <= xmin || ymax <= ymin; }
	BBox   intersect(BBox const &o) const;
	bool   contains(double x, double y) const { return x >= xmin && x < xmax && y >= ymin && y < ymax; }
};

constexpr int EPSG_LAMBERT72 = 31370;
constexpr int EPSG_WGS84     = 4326;
constexpr int EPSG_MERCATOR  = 3857;

// One-time GDAL/PROJ initialisation (drivers, resource paths). Safe to call several times.
void init();
// Releases GDAL/PROJ globals before process exit (avoids thread-pool hangs during DLL unload on Windows).
void shutdown();

// Lambert 72 <-> WGS 84 (PROJ, BD72 -> WGS84 Helmert, ~1 m)
LonLat toLonLat(XY lambert);
XY     toLambert(LonLat ll);

// Web Mercator (EPSG:3857) <-> WGS 84, closed form
XY     mercator(LonLat ll);
LonLat fromMercator(XY m);

// Slippy map tiles: fractional tile coordinates at zoom z (x east, y south from the top)
struct TileXY
{
	double x = 0, y = 0;
};
TileXY toTile(LonLat ll, int z);
LonLat fromTile(TileXY t, int z);

// Bounding box of a Lambert 72 box in another frame (hull of the 4 corners + edge midpoints)
BBox lambertToLonLatBox(BBox const &l72);
BBox lonLatToLambertBox(BBox const &ll);

// Generic transform (any EPSG -> any EPSG) for the WCS sources. Returns false on failure.
bool transform(int fromEpsg, int toEpsg, double &x, double &y);

// Bounding box of `box` (in fromEpsg) expressed in toEpsg: hull of corners and edge midpoints.
BBox transformBox(BBox const &box, int fromEpsg, int toEpsg);

// Working projection for an area: Belgian Lambert 72 around Belgium, else the WGS 84 / UTM zone of the point
// (EPSG:326xx north, 327xx south). All solver grids are metric and axis-aligned in that projection.
int suggestEpsg(LonLat centre);

// Human-readable name of a working projection ("Lambert 72", "UTM 33N")
std::string crsName(int epsg);

// Parses "50.63, 5.57", "50.63 5.57", a Google Maps URL (@lat,lon / ?q=lat,lon / ll=lat,lon) or an
// OpenStreetMap URL (#map=z/lat/lon). Returns false when nothing usable is found.
bool parseLocation(std::string const &text, LonLat &out);

}        // namespace swe::geo
