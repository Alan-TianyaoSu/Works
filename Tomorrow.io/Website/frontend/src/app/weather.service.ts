import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class WeatherService {
  private weatherDataSubject = new BehaviorSubject<any>(null);
  private locationDataSubject = new BehaviorSubject<any>(null);

  getWeatherData(): Observable<any> {
    return this.weatherDataSubject.asObservable();
  }

  setWeatherData(data: any): void {
    this.weatherDataSubject.next(data);
  }

  getLocationData(): Observable<any> {
    return this.locationDataSubject.asObservable();
  }

  setLocationData(data: any): void {
    this.locationDataSubject.next(data);
  }
}