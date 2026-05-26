"use client";

import React, { useEffect, useState, useCallback } from "react";
import { GoogleMap, useJsApiLoader, Marker, InfoWindow } from "@react-google-maps/api";

export interface ProviderMapProps {
  zipCodes: string[];
  names?: string[];
  coordinates?: Array<{ lat: number; lng: number; name?: string }>;
}

interface GeoLocation {
  lat: number;
  lng: number;
  name: string;
}

const containerStyle = {
  width: "100%",
  height: "100%",
  minHeight: "200px",
  borderRadius: "12px",
};

export default function Map({ zipCodes, names = [], coordinates }: ProviderMapProps) {
  const [locations, setLocations] = useState<GeoLocation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<GeoLocation | null>(null);

  const { isLoaded } = useJsApiLoader({
    googleMapsApiKey: process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY || "",
  });

  useEffect(() => {
    if (coordinates && coordinates.length > 0) {
      setLocations(
        coordinates.map((coord, index) => ({
          lat: coord.lat,
          lng: coord.lng,
          name: coord.name || names[index] || "Provider",
        }))
      );
      setError(null);
      setLoading(false);
      return;
    }

    async function fetchLocations() {
      const results: GeoLocation[] = [];

      for (let i = 0; i < zipCodes.length; i++) {
        try {
          const res = await fetch(`/api/geocode?zip=${zipCodes[i]}`);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();
          const first = Array.isArray(data) ? data[0] : null;
          if (first && first.lat != null && first.lon != null) {
            results.push({
              lat: parseFloat(first.lat),
              lng: parseFloat(first.lon),
              name: names?.[i] ?? "Provider",
            });
          }
        } catch (err) {
          console.warn(`Could not geocode ZIP ${zipCodes[i]}`, err);
        }
      }

      setLocations(results);
      if (results.length === 0) {
        setError("Unable to locate this provider on the map.");
      }
      setLoading(false);
    }

    if (zipCodes.length) {
      setLoading(true);
      fetchLocations();
    } else {
      setLocations([]);
      setLoading(false);
    }
  }, [zipCodes, names, coordinates]);

  const onLoad = useCallback(() => {}, []);

  if (loading || !isLoaded) {
    return (
      <div className="w-full h-full min-h-[200px] bg-gray-100 animate-pulse rounded-xl" />
    );
  }

  if (error || !locations.length) {
    return null;
  }

  const center = { lat: locations[0].lat, lng: locations[0].lng };

  return (
    <GoogleMap
      mapContainerStyle={containerStyle}
      center={center}
      zoom={15}
      onLoad={onLoad}
      options={{
        streetViewControl: false,
        mapTypeControl: false,
        fullscreenControl: false,
        zoomControlOptions: { position: 7 },
      }}
    >
      {locations.map((loc, idx) => (
        <Marker
          key={idx}
          position={{ lat: loc.lat, lng: loc.lng }}
          onClick={() => setSelected(loc)}
        />
      ))}
      {selected && (
        <InfoWindow
          position={{ lat: selected.lat, lng: selected.lng }}
          onCloseClick={() => setSelected(null)}
        >
          <div className="text-sm font-semibold text-gray-800">{selected.name}</div>
        </InfoWindow>
      )}
    </GoogleMap>
  );
}
