// src/app/geocoding.service.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

interface Geolocation {
  lat: number;
  lng: number;
}

@Injectable({
  providedIn: 'root'
})
export class GeocodingService {
  private googleApiKey = 'AIzaSyBOI4I20qLdWyWRU3ha2r_SErsdCqTkCRg';  

  constructor(private http: HttpClient) {}

  getGeolocation(street: string, city: string, state: string): Observable<Geolocation> {
    const formattedStreet = street.replace(/\s/g, '+');
    const formattedCity = city.replace(/\s/g, '+');
    const formattedState = state.replace(/\s/g, '+');
    
    const Add_String = [formattedStreet, formattedCity, formattedState]
      .filter(part => part) 
      .join('+');

    const apiUrl = `https://maps.googleapis.com/maps/api/geocode/json?address=${Add_String}&key=${this.googleApiKey}`;
    
    
    return this.http.get<any>(apiUrl).pipe(
      map(response => {
        if (response.results && response.results.length > 0) {
            const location = response.results[0].geometry.location;
            return { lat: location.lat, lng: location.lng };
        } else {
            throw new Error('No results found');
        }
      })
    );
  }
}