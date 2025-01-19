import { Component, Input, OnInit, ChangeDetectorRef, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common'; 
import * as Highcharts from 'highcharts';
import HighchartsMore from 'highcharts/highcharts-more';
import { HighchartsChartModule } from 'highcharts-angular';
import windbarb from 'highcharts/modules/windbarb';
import { trigger, transition, style, animate } from '@angular/animations';
import { Router } from '@angular/router';
import { GoogleMapComponent } from '../google-map/google-map.component';
import { WeatherService } from '../weather.service';
import { forkJoin } from 'rxjs';
import { firstValueFrom } from 'rxjs';  // 导入 firstValueFrom
import { HttpClient } from '@angular/common/http';   

interface LocationData {
  street: string;
  city?: string;  // 这里可以根据需要添加更多字段
  state?: string;
  latitude?: string;
  longitude?: string;
}

@Component({
  selector: 'app-weather-table',
  standalone: true,  
  templateUrl: './weather-table.component.html',
  styleUrls: ['./weather-table.component.css'],
  imports: [
    CommonModule,
    HighchartsChartModule,
    GoogleMapComponent,
  ],
  animations: [
    trigger('slideAnimation', [
      transition('results => details', [
        style({ transform: 'translateX(100%)' }), 
        animate('0.5s ease-out', style({ transform: 'translateX(0)' })) 
      ]),
      transition('details => results', [
        style({ transform: 'translateX(-100%)' }), 
        animate('0.5s ease-out', style({ transform: 'translateX(0)' })) 
      ]),
    ])
  ]
})

export class WeatherTableComponent implements OnInit {
  @Input() isFavorite: boolean = false;
  @Input() Page_id: string = '';
  @Output() favoriteChanged = new EventEmitter<{ isFavorite: boolean, pageId: string }>();
  currentDate: number = 0;
  Date_Selected: boolean = false;
  // isFavorite: boolean = false;
  displayedColumns: string[] = ['#', 'date', 'status', 'tempHigh', 'tempLow', 'windSpeed']; // 表格列
  currentPage: string = 'results';
  Map_Loaded = false;
  weatherData: any = [];
  locationData: LocationData;
  pageSize: number = 5;
  pagedData: any[] = [];
  currentView: string = 'dayView'; 
  dailyChartLib: typeof Highcharts = Highcharts;
  dailyChartOptions: Highcharts.Options | undefined;
  meteogramChartLib: typeof Highcharts = Highcharts;
  meteogramOptions: Highcharts.Options | undefined;
  Page: number = 1;

  X_Location: string = '';
  Show_Location: string = '';
  X_Date: string = '';
  X_Temperatue: number = 0;
  X_Condition: string = 'Clear';

  Summary_Weather: any[] = [];
  Daily_Weather: any[] = [];
  XRange_Data: any[] = [];
  Hourly_Weather: any[] = [];

  mapOptions: google.maps.MapOptions = {
    center: { lat: 34.0522, lng: -118.2437 },
    zoom: 12,
    mapTypeId: 'roadmap',
    zoomControl: true,
    scrollwheel: true,
    disableDoubleClickZoom: false,
  };
  map!: google.maps.Map; 

  // data =  [
  //   [Date.UTC(2023, 0, 1), -5, 10],  
  //   [Date.UTC(2023, 0, 2), -3, 12],  
  //   [Date.UTC(2023, 0, 3), 0, 14],   
  //   [Date.UTC(2023, 0, 4), -2, 8],   
  //   [Date.UTC(2023, 0, 5), -6, 9]    
  // ];

  // startDate = 'Feb 1st 2020';
  // endDate = 'Dec 31st 2020';

  constructor(private weatherService: WeatherService, private cdRef: ChangeDetectorRef, private http: HttpClient,) {
    this.locationData = {
      street: '123 Main St',
      city: 'YourCity',
      state: 'state',
      latitude: '123',
      longitude: '123',
    };
  }
  
  async ngOnInit(): Promise<void> {
    try {

      this.weatherData = await firstValueFrom(this.weatherService.getWeatherData());
      this.locationData = await firstValueFrom(this.weatherService.getLocationData());
      
      this.Data_Process();
      // this.updatePagedData();
      this.initializeHighcharts();
      this.processData();
      this.initMeteogram();
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  }

  async Change_to_date(day: number): Promise<void> {
    this.currentDate = day
    this.X_Process();
    this.showDetails()
  }

  Data_Process() {
    this.extractDetailWeather(this.weatherData['Data_1D']);
    this.extractHourlyWeather(this.weatherData['Data_1H']);
    this.X_Process();
  }

  X_Process = (): void => {
    this.X_Location = [this.locationData['street'], this.locationData['city'], this.locationData['state']]
    .filter(part => part)  
    .join(', ');
    this.Show_Location = [this.locationData['city'], this.locationData['state']]
    .filter(part => part)  
    .join(', ');
    this.X_Date = this.Summary_Weather[this.currentDate]['Date'];
    this.X_Temperatue = this.Summary_Weather[this.currentDate]['Temp'];
    this.X_Condition = this.Summary_Weather[this.currentDate]['Status'];
  };

  parseTemperature = (temp: string): number => parseFloat(temp.replace('°', ''));
  parseWindSpeed = (speed: string): number => parseFloat(speed.replace('mph', ''));
  formatTime = (isoString: string): string => {
    const date = new Date(isoString);
    const hours = date.getUTCHours();
    const minutes = date.getUTCMinutes();
    const ampm = hours >= 12 ? 'PM' : 'AM';
    const hour12 = hours % 12 === 0 ? 12 : hours % 12;
    return `${hour12}:${minutes.toString().padStart(2, '0')} ${ampm}`;
  };

  extractDetailWeather = (data: any): void => {
    const detailedWeather = [];
    const Xrange = [];
    const Summary = [];
    for (const day in data) {
      if (data.hasOwnProperty(day)) {
        const dayData = data[day];
        // console.log('Day: ', dayData.Weather)
        const details = [
          { label: 'Status', value: dayData.Weather },                              
          { label: 'Max Temperature', value: dayData.temperatureMax }, 
          { label: 'Min Temperature', value: dayData.temperatureMin }, 
          { label: 'Apparent Temperature', value: dayData.temperatureApparent },
          { label: 'Sun Rise Time', value: this.formatTime(dayData.sunriseTime) },       
          { label: 'Sun Set Time', value: this.formatTime(dayData.sunsetTime) },         
          { label: 'Humidity', value: dayData.humidity},                           
          { label: 'Wind Speed', value: dayData.windSpeed },       
          { label: 'Visibility', value: dayData.visibility },                       
          { label: 'Cloud Cover', value: dayData.cloudcover },                            
        ];
        const xrange = [
          new Date(dayData.Time).getTime(),
          this.parseTemperature(dayData.temperatureMin),
          this.parseTemperature(dayData.temperatureMax),
        ];
        const Day_Sum = {
          Date: dayData.Time,
          Images: dayData.Image,
          Status: dayData.Weather,
          Temp_High: this.parseTemperature(dayData.temperatureMax),
          Temp_Low: this.parseTemperature(dayData.temperatureMin),
          Wind_Speed: this.parseWindSpeed(dayData.windSpeed),
          Temp: this.parseTemperature(dayData.temperatureApparent),
        };
        detailedWeather.push(details);
        Xrange.push(xrange);
        Summary.push(Day_Sum);
      }
    }
    this.Daily_Weather = detailedWeather;
    
    this.XRange_Data = Xrange;
    this.Summary_Weather = Summary;
    console.log('XRange_Data: ', this.XRange_Data)
  };

  extractHourlyWeather = (data: any): void => {
    const Hourly = [];
    for (const day in data) {
      if (data.hasOwnProperty(day)) {
        const dayData = data[day];
        const Hourly_data = {
          time: dayData.time,
          temperature: dayData.temperature,
          humidity: dayData.humidity,
          wind_speed: dayData.wind_speed,
          wind_from_direction: dayData.wind_from_direction,
          air_pressure_at_sea_level: dayData.air_pressure_at_sea_level,
        };
        Hourly.push(Hourly_data)
      }
    }
    this.Hourly_Weather = Hourly
  };

  goToX() {
    const tweetText = encodeURIComponent(`The temperature in ${this.X_Location} on ${this.X_Date} is ${this.X_Temperatue}°F and the conditions are ${this.X_Condition} #CSCI571WeatherForecast`);
    const tweetUrl = `https://x.com/intent/tweet?text=${tweetText}`;
  
    window.open(tweetUrl, '_blank');
  }

  async toggleFavorite(): Promise<void> {
    this.isFavorite = !this.isFavorite;
    console.log("Try to Emit PageID: ", this.Page_id)
    if (this.isFavorite) {
      await this.Item_Insert();
    } else {
      await this.Item_Delete();
    }
    console.log("Emit PageID: ", this.Page_id)
    this.favoriteChanged.emit({ isFavorite: this.isFavorite, pageId: this.Page_id });
  }

  Item_Insert(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      console.log('Inserting item:', this.locationData);
      
      this.http.post<{ message: string, insertedId: string }>(`https://assignment3-784518.wl.r.appspot.com/Insert`, this.locationData)
        .subscribe(
          (response) => {
            console.log('Item inserted successfully', response.insertedId);
            this.Page_id = response.insertedId;
            resolve();  // 插入成功，解决 promise
          },
          (error) => {
            console.error('Error inserting item', error);
            reject(error);  // 插入失败，拒绝 promise
          }
        );
    });
  }

  Item_Delete(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      console.log('No longer like :', this.Page_id);
      
      // 发起 HTTP DELETE 请求
      this.http.delete(`https://assignment3-784518.wl.r.appspot.com/Delete/${this.Page_id}`)
        .subscribe(
          () => {
            this.Page_id = '';  // 删除成功后，清空 InsertID
            console.log('Item deleted successfully');
            resolve();  // 删除成功，解决 promise
          },
          (error) => {
            console.error('Error deleting item', error);
            reject(error);  // 删除失败，拒绝 promise
          }
        );
    });
  }

  setView(page: string) {
    this.currentView = page;
  }
  
  showDetails() {
    this.currentPage = 'details';
    this.cdRef.detectChanges();
    this.initializeMap();
    this.changeMapCenterAndZoom()
  }

  showResults() {
    this.currentPage = 'results';
  }

  Set_Map_Option(newOptions: google.maps.MapOptions): void {
    if (this.map) {
      this.map.setOptions(newOptions);
    } else {
      console.error('Map is not initialized yet.');
    }
  }

  changeMapCenterAndZoom(): void {
    const newOptions: google.maps.MapOptions = {
      center: { lat: Number(this.locationData['latitude']), lng: Number(this.locationData['longitude']) },
      zoom: 14,
    };
    console.log('New Center: ', newOptions['center'])
    this.Set_Map_Option(newOptions);
  }

  initializeMap(): void {
    const mapContainer = document.getElementById('map') as HTMLElement;
    this.map = new google.maps.Map(mapContainer, this.mapOptions);
  }

  initializeHighcharts(): void {
    this.dailyChartLib = Highcharts;
    HighchartsMore(Highcharts);
    this.dailyChartOptions = {
      chart: {
        type: 'arearange',
        zooming: {
          type: 'x'
        },
        width: null,
        height: this.vhToPx(40),
      },
      title: {
        text: 'Temperature Ranges (Min, Max)'
      },
      xAxis: {
        type: 'datetime',
        // accessibility: {
        //   rangeDescription: `Range: ${this.startDate} to ${this.endDate}.`
        // },
      },
      yAxis: {
        title: {
          text: null
        },
        crosshair: {
          color: '#00ff00',  
          width: 1,  
          dashStyle: 'Dot'  
        }
      },
      tooltip: {
        shared: true,
        valueSuffix: '°F',
        xDateFormat: '%A, %b %e',
      },
      legend: {
        enabled: false
      },
      series: [{
        type: 'arearange',
        name: 'Temperatures',
        data: this.XRange_Data,
        lineWidth: 1.5,            
        color: {
          linearGradient: {
            x1: 0,
            x2: 0,
            y1: 0,
            y2: 1
          },
          stops: [
            [0, '#f1a733'],
            [1, '#d0cdbe']
          ]
        },
        lineColor: '#45a1d8',
        marker: {
          enabled: true,          
          fillColor: '#45a1d8',   
          lineWidth: 2,           
          // lineColor: '#ffffff',   
          radius: 6  
        }
      }]
    }
  }
  
  

  vwToPx(vw: number): number {
    return Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0) * vw / 100;
  }

  vhToPx(vh: number): number {
    return Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0) * vh / 100;
  }

  temperatures: { x: number, y: number }[] = [];
  humidity: { x: number, y: number }[] = [];
  winds: { x: number, value: number, direction: number }[] = [];
  pressures: { x: number, y: number }[] = [];
  
  processData(): void {
    this.Hourly_Weather.forEach((node, i) => {
      const x = Date.parse(node.time); 
      this.temperatures.push({
        x,
        y: Math.round(node.temperature) 
      });
      this.humidity.push({
        x,
        y: Math.round(node.humidity) 
      });
      if (i % 2 === 0) {
        this.winds.push({
          x,
          value: Math.round(node.wind_speed * 100) / 100, 
          direction: node.wind_from_direction 
        });
      }
      this.pressures.push({
        x,
        y: Math.round(node.air_pressure_at_sea_level)
      });
    });
  }

  initMeteogram(): void {
    this.meteogramChartLib = Highcharts;
    HighchartsMore(Highcharts);
    windbarb(Highcharts);
    this.meteogramOptions = {
      chart: {
          marginBottom: 70,
          marginRight: 40,
          marginTop: 50,
          plotBorderWidth: 1,
          width: null,
          height: this.vhToPx(40),
          alignTicks: false,
      },

      title: {
          text: 'Hourly Weather(For next 5 Days)',
          align: 'center',
          style: {
              whiteSpace: 'nowrap',
              textOverflow: 'ellipsis'
          }
      },
      tooltip: {
          shared: true,
          useHTML: true,
          headerFormat:
              '<small>{point.x:%A, %b %e, %H:%M} - ' +
              '{point.point.to:%H:%M}</small><br>' +
              '<b>{point.point.symbolName}</b><br>'
      },

      xAxis: [{ // Bottom X axis
          type: 'datetime',
          // accessibility: {
          //   rangeDescription: `Range: ${this.startDate} to ${this.endDate}.`
          // },
          tickInterval: 2 * 36e5, // two hours
          minorTickInterval: 36e5, // one hour
          tickLength: 0,
          gridLineWidth: 1,
          gridLineColor: 'rgba(128, 128, 128, 0.1)',
          startOnTick: false,
          endOnTick: false,
          minPadding: 0,
          maxPadding: 0,
          offset: 30,
          showLastLabel: true,
          labels: {
              format: '{value:%H}'
          },
          crosshair: true
      }, { // Top X axis
          linkedTo: 0,
          type: 'datetime',
          tickInterval: 24 * 3600 * 1000, // one day
          labels: {
              format: '{value:<span style="font-size: 12px; font-weight: ' +
                  'bold">%a</span> %b %e}',
              align: 'left',
              x: 3,
              y: 8
          },
          opposite: true,
          tickLength: 20,
          gridLineWidth: 1
      }],

      yAxis: [{ // temperature axis
          title: {
              text: null
          },
          labels: {
              format: '{value}°',
              style: {
                  fontSize: '10px'
              },
              x: -3
          },
          plotLines: [{ // zero plane
              value: 0,
              color: '#BBBBBB',
              width: 1,
              zIndex: 2
          }],
          maxPadding: 0.3,
          minRange: 8,
          tickInterval: 1,
          gridLineColor: 'rgba(128, 128, 128, 0.1)'
      }, { // precipitation axis
          title: {
              text: null
          },
          labels: {
              enabled: false
          },
          gridLineWidth: 0,
          tickLength: 0,
          minRange: 10,
          min: 0
      }, { // Air pressure
          allowDecimals: false,
          title: { // Title on top of axis
              text: 'inHg',
              offset: 0,
              align: 'high',
              rotation: 0,
              style: {
                  fontSize: '10px',
                  color: '#e4d354'
              },
              textAlign: 'left',
              x: 3
          },
          labels: {
              style: {
                  fontSize: '8px',
                  color: '#e4d354'
              },
              y: 2,
              x: 3
          },
          gridLineWidth: 0,
          opposite: true,
          showLastLabel: false
      }],

      legend: {
          enabled: false
      },

      plotOptions: {
          series: {
              pointPlacement: 'between'
          }
      },

      series: [{
          name: 'Temperature',
          data: this.temperatures,
          type: 'spline',
          marker: {
              enabled: false,
              states: {
                  hover: {
                      enabled: true
                  }
              }
          },
          tooltip: {
              pointFormat: '<span style="color:{point.color}">\u25CF</span>' +
                  ' ' +
                  '{series.name}: <b>{point.y}°F</b><br/>'
          },
          zIndex: 1,
          color: '#FF3333',
          negativeColor: '#48AFE8'
      },
      {
          name: 'Moisture',
          data: this.humidity,
          type: 'column',
          color: '#68CFE8',
          yAxis: 1,
          groupPadding: 0,
          pointPadding: 0,
          grouping: false,
          dataLabels: {
              enabled: true,
              filter: {
                  operator: '>',
                  property: 'y',
                  value: 0
              },
              style: {
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#666'
              }
          },
          tooltip: {
              valueSuffix: ' mm'
          }
      }, 
      {
        type: 'line',  
        name: 'Air pressure',
          color: '#e4d354',
          data: this.pressures,
          marker: {
              enabled: false
          },
          shadow: false,
          tooltip: {
              valueSuffix: ' inHg'
          },
          dashStyle: 'ShortDot',
          yAxis: 2
      }, 
      {
          name: 'Wind',
          type: 'windbarb',
          id: 'windbarbs',
          color: '#434348',
          lineWidth: 1.5,
          data: this.winds,
          vectorLength: 18,
          yOffset: -15,
          tooltip: {
              valueSuffix: ' m/s'
          }
      }]
  };
}
}
