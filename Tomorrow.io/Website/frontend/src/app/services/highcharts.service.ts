import { Injectable } from '@angular/core';
import * as Highcharts from 'highcharts';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class HighchartsService {

  constructor(private http: HttpClient) {}

  // Fetch data and configure the chart
  async createChart(containerId: string, query: string): Promise<void> {
    const element = document.getElementById('Temp_Range') as HTMLElement;

    if (!element) {
      console.error(`Element with id '${containerId}' not found`);
      return;
    }

    try {
      const storedQueryString = JSON.parse(query);

      // Fetch the data from the server
      const fetched: any = await firstValueFrom(
        this.http.get(`/Area_Weather?${storedQueryString}`, { responseType: 'text' })
      );
      const data = JSON.parse(fetched);

      // Highcharts configuration
      Highcharts.chart(element as HTMLElement, {
        chart: {
          type: 'arearange',
          zooming: {
            type: 'x'
          },
          width: this.vwToPx(50),
          height: this.vhToPx(40),
          scrollablePlotArea: {
            minWidth: 600,
            scrollPositionX: 1
          }
        },
        title: {
          text: 'Temperature Ranges (Min, Max)'
        },
        xAxis: {
          type: 'datetime',
          accessibility: {
            rangeDescription: 'Range: Jan 1st 2017 to Dec 31 2017.'
          }
        },
        yAxis: {
          title: {
            text: null
          }
        },
        tooltip: {
          crosshairs: true,
          shared: true,
          valueSuffix: '°F',
          xDateFormat: '%A, %b %e'
        },
        legend: {
          enabled: false
        },
        series: [{
          name: 'Temperatures',
          data: data,
          color: {
            linearGradient: {
              x1: 0,
              x2: 0,
              y1: 0,
              y2: 1
            },
            stops: [
              [0, '#f7a35c'],
              [1, '#7cb5ec']
            ]
          },
          lineColor: '#f7a35c',
          marker: {
            enabled: true,
            fillColor: '#7cb5ec',
            lineWidth: 2,
            lineColor: '#7cb5ec',
            radius: 2
          }
        }]
      });
    } catch (error) {
      console.error('Error fetching or processing data:', error);
    }
  }

  // Utility functions to convert vw and vh to pixels
  vwToPx(vw: number): number {
    return Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0) * vw / 100;
  }

  vhToPx(vh: number): number {
    return Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0) * vh / 100;
  }
}