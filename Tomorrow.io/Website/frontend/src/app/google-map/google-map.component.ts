import { Component, AfterViewInit } from '@angular/core';
import { GoogleMapsModule } from '@angular/google-maps';

@Component({
  selector: 'app-google-map',
  standalone: true,
  imports: [
    GoogleMapsModule,
  ],
  templateUrl: './google-map.component.html',
  styleUrl: './google-map.component.css'
})

export class GoogleMapComponent{

  center: google.maps.LatLngLiteral = { lat: 34.0522, lng: -118.2437 }; // Example: Los Angeles
  zoom = 12;
  
}
