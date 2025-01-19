import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';  
import { RouterOutlet } from '@angular/router';
import { FormControl, FormsModule } from '@angular/forms';
import { GeocodingService } from './geocoding.service';
import { HttpClient } from '@angular/common/http';      
import { Observable } from 'rxjs';
import { lastValueFrom } from 'rxjs';
import { WeatherService } from './weather.service';
import { WeatherTableComponent } from './weather-table/weather-table.component';
import {MatAutocompleteModule} from '@angular/material/autocomplete';
import {MatInputModule} from '@angular/material/input';
import {MatFormFieldModule} from '@angular/material/form-field';
import { debounceTime, switchMap, map, catchError, startWith } from 'rxjs/operators';
import {AsyncPipe} from '@angular/common';
import {ReactiveFormsModule} from '@angular/forms';

interface IpInfo {
  city: string;
  region: string;
  loc: string;
}

interface FormData {
  street: string;
  city: string;
  state: string;
  latitude: Number | null;
  longitude: Number | null;
}

interface AutocompletePrediction {
  city: string;
  state: string;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    RouterOutlet, 
    CommonModule,
    FormsModule,
    WeatherTableComponent,
    MatFormFieldModule,
    MatInputModule,
    MatAutocompleteModule,
    AsyncPipe,
    ReactiveFormsModule,
    // BrowserModule,
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',

})

export class AppComponent implements OnInit {
  title = 'frontend';
  selectedButton: string = 'results'; 

  street: string = '';
  city: string = '';
  state: string = '';

  streetTouched: boolean = false;
  cityTouched: boolean = false;
  stateTouched: boolean = false;
  currentLocation: boolean = false; 
  showTable: boolean = false;  
  formData: any = {};
  GeoData: any = {};
  weatherData: any = {};  
  Data_Base: any[] = [];
  isFavorite: boolean = false;
  Page_id: string = '';

  loading: boolean = false; 
  progress: number = 0;    

  constructor(private geocodingService: GeocodingService, private http: HttpClient, private weatherService: WeatherService) {}

  stateControl = new FormControl('');
  cityControl = new FormControl('');

  filteredCities: Observable<string[]> = new Observable<string[]>();
  filteredStates: Observable<string[]> = new Observable<string[]>();

  // states: string[] = ['California', 'Texas', 'Florida', 'New York', 'Illinois'];
  filteredCityOptions: AutocompletePrediction[] = [];
  filteredStateOptions: string[] = [];

  private googleApiKey = 'AIzaSyBOI4I20qLdWyWRU3ha2r_SErsdCqTkCRg'; // Replace with your Google API Key
  private autocompleteUrl = 'https://maps.googleapis.com/maps/api/place/autocomplete/json';

  ngOnInit(): void {
    this.filteredStateOptions = this.states;
  }

  onCityInput(event: Event): void {
    
    const inputValue = (event.target as HTMLInputElement).value;
    
    if (inputValue.trim().length > 0) {
      this.fetchCities(inputValue).subscribe(predictions => {
        this.filteredCityOptions = predictions
      });
    } else {
      this.filteredCityOptions = [];
    }
  }
  
  private apiUrl = 'https://assignment3-784518.wl.r.appspot.com/Get_autocomplete';
  fetchCities(input: string): Observable<any[]> {
    const url = `${this.apiUrl}?input=${input}`;
    return this.http.get<{ predictions: any[] }>(url).pipe(
      map(response => {
        return response.predictions.map(prediction => {
          const city = prediction.terms[0]?.value || '';
          const state = prediction.terms[1]?.value || '';
          return { city, state };
        });
      })
    );
  }

  onStateInput(event: Event): void {
    const inputValue = (event.target as HTMLInputElement).value.toLowerCase();
    this.filteredStateOptions = this.states.filter(state =>
      state.toLowerCase().includes(inputValue)
    );
  }

  getCityPredictions(input: string): Observable<AutocompletePrediction[]> {
    const params = {
      input,
      key: this.googleApiKey,
      types: 'geocode' 
    };

    return this.http.get<{ predictions: AutocompletePrediction[] }>(this.autocompleteUrl, { params })
      .pipe(map(response => response.predictions));
  }


  

  selectButton(button: string) {
    this.selectedButton = button;
    this.errorMessage = false;
    if(button === 'favorites'){
      this.Get_DataBase();
    }
  }
  
  async onSubmit(form: any): Promise<void> {
  
    this.half_clearForm();
    this.loading = true;
    this.progress = 0;
    this.simulateLoading();
    if (form.valid) {
      this.formData = form.value;
      this.GeoData = await this.onSearch();
      if (this.errorMessage) {
        this.loading = false;
        this.progress = 0;
      }
      else{
        this.sendDataToBackend().subscribe(
          (data) => {
            this.weatherData = data;
            this.weatherService.setWeatherData(this.weatherData);
            this.weatherService.setLocationData(this.GeoData);
            this.showTable = true
            this.loading = false;
            this.progress = 0;
          },
          (error) => {
            this.errorMessage = true;
            this.showTable = false
            this.loading = false;
            this.progress = 0;
            console.error('Subscribe Error:', error);
          }
        );
      }
      
    } else {
      console.error('Form is invalid');
      this.loading = false;
      this.progress = 0;
    }
  }

  simulateLoading() {
    const interval = setInterval(() => {
      this.progress += 10; 
      if (this.progress >= 100) {
        clearInterval(interval);
        this.loading = false; 
      }
    }, 500); 
  }

  Get_DataBase(): void {
    this.getDataFromBackend().subscribe(
      (data) => {
        this.Data_Base = data;
      }
    );
  }
  
  sendDataToBackend(): Observable<any> {
    const backendUrl = `https://assignment3-784518.wl.r.appspot.com/Get_Weather?longitude=${this.GeoData.longitude}&latitude=${this.GeoData.latitude}`;
    // "https://api.tomorrow.io/v4/timelines?location=34.0522,-118.2437&fields=weatherCode,temperature,temperatureMax,temperatureMin,precipitationType,precipitationProbability,windSpeed,humidity,visibility,sunriseTime&timesteps=1d&units=imperial&timezone=America/Los_Angeles&apikey=TS1FKJFkwAECNDQ4V8D9ZCpIevYPwC4G"
    return this.http.get(backendUrl);
  }

  getDataFromBackend(): Observable<any> {
    const DBUrl = `https://assignment3-784518.wl.r.appspot.com/Get_Data`;
    return this.http.get(DBUrl);
  }

  clearForm(form: any) {
    form.resetForm();  
    this.streetTouched = false;
    this.cityTouched = false;
    this.stateTouched = false;
    this.showTable = false;
    this.isFavorite = false;
    this.Page_id = '';
    this.errorMessage = false;
    this.selectedButton = 'results';

  }

  half_clearForm() {
    // form.resetForm();  
    this.streetTouched = false;
    this.cityTouched = false;
    this.stateTouched = false;
    this.showTable = false;
    this.isFavorite = false;
    this.Page_id = '';
    this.errorMessage = false;
  }

  onSearch() {
    if (this.currentLocation) {
      return this.Perform_Autocheck();
    }
    return this.Perform_Certain_Check();
  }

  latitude: Number | null = null;
  longitude: Number | null = null;
  errorMessage: boolean = false;
  Received_City: string = '';
  Received_state: string = '';

  async Perform_Autocheck() {
    try{
    const response = await fetch(`https://ipinfo.io?token=dc9d2d418e567f`);
    const In_Info: IpInfo = await response.json();
    [this.latitude, this.longitude] = In_Info.loc.split(',').map(Number);
    this.Received_City = In_Info.city;
    this.Received_state = In_Info.region;
    }catch (error) {
      this.latitude = null;
      this.longitude = null;
      this.Received_City = '';
      this.Received_state = '';
      this.errorMessage = true;
    }
    const formData: FormData = {
      street: '',
      city: this.Received_City,
      state: this.Received_state,
      latitude: this.latitude,
      longitude: this.longitude,
    };
    return formData
  }
  
  async Perform_Certain_Check(): Promise<FormData> {
    const street = this.formData['street'];
    const city = this.formData['city'];
    const state = this.formData['state'];
    this.errorMessage = false;
    try {
      const location = await lastValueFrom(this.geocodingService.getGeolocation(street, city, state));
      this.latitude = location.lat;
      this.longitude = location.lng;
      this.errorMessage = false;
    } catch (error) {
      this.latitude = null;
      this.longitude = null;
      this.errorMessage = true;
    }
  
    const formData: FormData = {
      street: street,
      city: city,
      state: state,
      latitude: this.latitude,
      longitude: this.longitude,
    };
    return formData;
    
  }

  handleClick(item: any): void {
    this.loading = true;
    this.progress = 0;
    this.simulateLoading();
    this.GeoData = {
      street: item.street,
      city: item.city,
      state: item.state,
      latitude: item.latitude,
      longitude: item.longitude,
    };
    this.sendDataToBackend().subscribe(
      (data) => {
        this.weatherData = data;
        this.weatherService.setWeatherData(this.weatherData);
        this.weatherService.setLocationData(this.GeoData);
        this.showTable = true;
        this.selectedButton = 'results';
        this.isFavorite = true;
        this.Page_id = item._id;
        this.loading = false;
        this.progress = 0;
      }
    );
  }

  handleDelete(item: any): void {
    
    this.http.delete(`https://assignment3-784518.wl.r.appspot.com/Delete/${item._id}`)
      .subscribe(
        () => {
          this.Data_Base = this.Data_Base.filter(data => data._id !== item._id);
          
          if (this.Page_id == item._id){
            this.isFavorite = false;
            this.Page_id = '';
          }
        },
        (error) => {
          console.error('Error deleting item', error);
        }
      );
  }

  onFavoriteChanged(event: { isFavorite: boolean, pageId: string }): void {
    this.isFavorite = event.isFavorite; 
    this.Page_id = event.pageId; 
  }





states: string[] = [
  'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
  'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
  'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
  'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
  'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma',
  'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas',
  'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
];


// cities: string[] = [
//   'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio',
//   'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville', 'Fort Worth', 'Columbus',
//   'Charlotte', 'San Francisco', 'Indianapolis', 'Seattle', 'Denver', 'Washington'
// ];
}
