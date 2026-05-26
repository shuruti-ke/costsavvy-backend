"use client";

import React, { useEffect } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet-defaulticon-compatibility";
import "leaflet-defaulticon-compatibility/dist/leaflet-defaulticon-compatibility.css";
import L from "leaflet";

interface MapFacility {
  list_index: number;
  facility_key: string;
  name: string;
  latitude: number;
  longitude: number;
  address: string;
  price: number | null;
  estimated_range: string | null;
  website_url: string | null;
  price_source?: string | null;
  insurance_match?: boolean;
  insurance_provider_name?: string | null;
  matching_insurers?: string[];
}

interface MapData {
  center: { latitude: number; longitude: number };
  user_location: { latitude: number; longitude: number; zipcode: string };
  searched_insurance?: string;
  facilities: MapFacility[];
  google_maps_url?: string;
}

interface Props {
  mapData: MapData;
  selectedIdx: number | null;
  onSelectFacility: (i: number) => void;
}

function FitBounds({ mapData }: { mapData: MapData }) {
  const map = useMap();
  useEffect(() => {
    const pts: [number, number][] = [[mapData.user_location.latitude, mapData.user_location.longitude]];
    mapData.facilities.forEach(f => pts.push([f.latitude, f.longitude]));
    if (pts.length > 1) map.fitBounds(pts, { padding: [40, 40], maxZoom: 14 });
    else map.setView([mapData.user_location.latitude, mapData.user_location.longitude], 12);
  }, [mapData, map]);
  return null;
}

const userIcon = L.divIcon({
  className: "",
  html: `<div style="width:18px;height:18px;border-radius:50%;background:#2563eb;border:3px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.3)"></div>`,
  iconSize: [18, 18],
  iconAnchor: [9, 9],
});

function facilityIcon(num: number, insuranceMatch = false, selected = false) {
  const bg = insuranceMatch ? "#15803d" : "#6b2458";
  const ring = selected ? "0 0 0 4px rgba(255,255,255,0.75), 0 0 0 7px rgba(107,36,88,0.18)" : "0 2px 6px rgba(0,0,0,0.3)";
  return L.divIcon({
    className: "",
    html: `<div style="width:${selected ? 34 : 28}px;height:${selected ? 34 : 28}px;border-radius:50%;background:${bg};color:white;border:2px solid white;box-shadow:${ring};display:flex;align-items:center;justify-content:center;font-weight:700;font-size:${selected ? 13 : 12}px;">${num}</div>`,
    iconSize: [selected ? 34 : 28, selected ? 34 : 28],
    iconAnchor: [selected ? 17 : 14, selected ? 17 : 14],
  });
}

export default function PriceSearchMap({ mapData, selectedIdx, onSelectFacility }: Props) {
  return (
    <MapContainer
      center={[mapData.center.latitude, mapData.center.longitude]}
      zoom={12}
      style={{ height: "100%", minHeight: 440, width: "100%" }}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <FitBounds mapData={mapData} />

      {/* User marker */}
      <Marker position={[mapData.user_location.latitude, mapData.user_location.longitude]} icon={userIcon}>
        <Popup><strong>Your Location</strong><br />{mapData.user_location.zipcode}</Popup>
      </Marker>

      {/* Facility markers */}
      {mapData.facilities.map((f, i) => (
        <Marker
          key={f.facility_key || i}
          position={[f.latitude, f.longitude]}
          icon={facilityIcon(f.list_index, Boolean(f.insurance_match), selectedIdx === i)}
          eventHandlers={{ click: () => onSelectFacility(i) }}
        >
          <Popup>
            <div style={{ minWidth: 180 }}>
              <p style={{ fontWeight: 600, color: "#6b2458", fontSize: 14, marginBottom: 4 }}>{f.name}</p>
              {f.address && <p style={{ fontSize: 12, color: "#666", marginBottom: 6 }}>{f.address}</p>}
              <p style={{ fontWeight: 700, fontSize: 15, color: f.price ? "#059669" : "#888" }}>
                {f.price ? `$${f.price.toLocaleString()}` : f.estimated_range ? `${f.estimated_range} (est.)` : "Contact for pricing"}
              </p>
              <p style={{ fontSize: 12, color: "#666", marginTop: 4 }}>
                {f.price_source === "web_search" || f.price_source === "web_candidate"
                  ? "Hospital-linked estimate page"
                  : "Healthcare facility"}
              </p>
              {(() => {
                const matchingInsurers = Array.from(
                  new Set((f.matching_insurers || []).map((insurer) => insurer.trim()).filter(Boolean))
                );
                const insurerLabel = matchingInsurers.length > 0
                  ? matchingInsurers.length === 1
                    ? matchingInsurers[0]
                    : matchingInsurers.join(", ")
                  : String(f.insurance_provider_name || "").trim();
                const displayInsurerLabel =
                  mapData.searched_insurance && insurerLabel && insurerLabel.toLowerCase() === mapData.searched_insurance.toLowerCase()
                    ? ""
                    : insurerLabel;
                const nearbyLabel = mapData.searched_insurance
                  ? displayInsurerLabel
                    ? `Nearby match: no ${mapData.searched_insurance} match; shown with ${displayInsurerLabel}`
                    : `Nearby match: no ${mapData.searched_insurance} match`
                  : displayInsurerLabel
                    ? `Nearby match: ${displayInsurerLabel}`
                    : "Nearby match";
                return (
                  <p style={{
                    display: "inline-flex",
                    marginTop: 8,
                    padding: "2px 8px",
                    borderRadius: 9999,
                    background: f.insurance_match ? "#ecfdf5" : "#fdf2f8",
                    color: f.insurance_match ? "#047857" : "#6b2458",
                    fontSize: 11,
                    fontWeight: 700,
                  }}>
                    {f.insurance_match
                      ? displayInsurerLabel
                        ? `Insurance match: ${displayInsurerLabel}`
                        : "Insurance match"
                      : nearbyLabel}
                  </p>
                );
              })()}
              {f.website_url && (
                <a href={f.website_url} target="_blank" rel="noopener noreferrer" style={{ fontSize: 12, color: "#2563eb", display: "block", marginTop: 8 }}>
                  {f.website_url.includes("google.com/maps") ? "View on Google Maps ↗" : "Visit website ↗"}
                </a>
              )}
            </div>
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}
