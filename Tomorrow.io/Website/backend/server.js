const express = require('express');
const { MongoClient } = require('mongodb');
const cors = require('cors');
const path = require('path');
const fetch = require('node-fetch');
const Weather_Mapping = require('./weather-mapping.js'); 

const PORT = 8080;
const app = express();

app.use(express.json()); 
app.use(cors());

// MongoDB URI
const uri = 'mongodb+srv://alantianyao:Sty20020407784518!@assignment3.ro4k8.mongodb.net/?retryWrites=true&w=majority&appName=Assignment3';

// Tomorrow API
const app_key = "TS1FKJFkwAECNDQ4V8D9ZCpIevYPwC4G"
// const app_key = "rRy4klckCQk2bL8ijgtBbfvrhuJpuDzV"
// const app_key = "iWKlQCy58YuZp47ZT3g2BRC9GDfuvrKe"
// const app_key = "ja0wA1hYa6JNQI6RxNo27x7egPp8rUTT"


const client = new MongoClient(uri, { useNewUrlParser: true, useUnifiedTopology: true });
const { ObjectId } = require('mongodb');  

async function connectToMongo() {
  try {
    await client.connect();
    console.log('Connected to MongoDB');
  } catch (error) {
    console.error('MongoDB connection error:', error);
  }
}
connectToMongo();

app.get('/', (req, res) => {
  console.log("Successfully connected to backend!")
});

app.get('/Get_Data', async (req, res) => {
  try {
    const database = client.db('Ass3Database');
    const collection = database.collection('Ass3Collection');
    const data = await collection.find({}).toArray();
    res.json(data);
  } catch (error) {
    console.error('Error fetching data from MongoDB:', error);
    res.status(500).send('Error fetching data from MongoDB');
  }
});

app.post('/Insert', async (req, res) => {
  try {
    const database = client.db('Ass3Database');
    const collection = database.collection('Ass3Collection');
    const newData = req.body;
    const result = await collection.insertOne(newData);
    res.status(201).json({ message: 'Document inserted', insertedId: result.insertedId });
  } catch (error) {
    console.error('Error inserting data into MongoDB:', error);
    res.status(500).send('Error inserting data into MongoDB');
  }
});

app.delete('/Delete/:id', async (req, res) => {
  try {
    const database = client.db('Ass3Database');
    const collection = database.collection('Ass3Collection');
    const documentId = req.params.id;
    const result = await collection.deleteOne({ _id: new ObjectId(documentId) });
    if (result.deletedCount === 1) {
      res.status(200).json({ message: 'Document deleted successfully' });
    } else {
      res.status(404).json({ message: 'Document not found' });
    }
  } catch (error) {
    console.error('Error deleting document from MongoDB:', error);
    res.status(500).send('Error deleting document from MongoDB');
  }
});

function Data_Process_1D (Json_1D) {
  const weather_data = Json_1D.data.timelines[0].intervals;
  const result_data = {};
  weather_data.forEach((interval, idx) => {
      const Inter = interval.values;
      const Cur_weather_Code = Inter.weatherCode;

      const startTime = new Date(interval.startTime);

      const options = { weekday: 'long', year: 'numeric', month: 'short', day: 'numeric' };
      const formattedTime = startTime.toLocaleDateString('en-US', options);

      const row = {
          'Time': formattedTime,
          'humidity': `${Inter.humidity}%`,
          'precipitationProbability': `${Inter.precipitationProbability}%`,
          'precipitationType': `${Inter.precipitationType}`,
          'sunriseTime': `${Inter.sunriseTime}`,
          'sunsetTime': `${Inter.sunsetTime}`,
          'temperatureApparent': `${parseFloat(Inter.temperatureApparent).toFixed(1).replace(/\.0$/, '')}°`,
          'temperatureMax': `${parseFloat(Inter.temperatureMax).toFixed(1).replace(/\.0$/, '')}°`,
          'temperatureMin': `${parseFloat(Inter.temperatureMin).toFixed(1).replace(/\.0$/, '')}°`,
          'visibility': `${Inter.visibility}mi`,
          'Image': `/${Weather_Mapping.Weather_Image_Mapping[Cur_weather_Code]}`,
          'Weather': Weather_Mapping.Weather_Name_Mapping[Cur_weather_Code],
          'windSpeed': `${Inter.windSpeed}mph`,
          'cloudcover': `${Inter.cloudCover}%`
      };
      const day_key = `day${idx + 1}`;
      result_data[day_key] = row;
  });
  return result_data;
}

function Data_Process_1H (Json_1H) {
  const output = [];
  const timelines = (Json_1H.data && Json_1H.data.timelines) || [];
  timelines.forEach(timeline => {
      const intervals = timeline.intervals || [];

      intervals.forEach(interval => {
          const start_time = interval.startTime || null;
          const values = interval.values || {};
          const new_entry = {
              time: start_time,
              temperature: values.temperature || null,
              humidity: values.humidity || null,
              wind_speed: values.windSpeed || null,
              wind_from_direction: values.windDirection || null,
              air_pressure_at_sea_level: values.pressureSeaLevel || null
          };
          output.push(new_entry);
      });
  });
  return output;
}

app.get('/Get_Weather', async (req, res) => {
  try {
    const { longitude, latitude } = req.query;
    if (longitude === undefined || latitude === undefined) {
      return res.status(400).json({ message: 'Missing longitude or latitude' });
    }
    // const Fields_1D = "weatherCode,temperature,temperatureMax,temperatureMin,precipitationType,precipitationProbability,windSpeed,humidity,visibility,sunriseTime,sunsetTime"
    const Fields_1D = "weatherCode,temperatureApparent,temperatureMax,temperatureMin,precipitationType,precipitationProbability,windSpeed,humidity,visibility,sunriseTime,sunsetTime,cloudCover"
    const Fields_1H = "temperature,humidity,windSpeed,windDirection,pressureSeaLevel"
    const Tomo_API_1D = `https://api.tomorrow.io/v4/timelines?location=${latitude},${longitude}&fields=${Fields_1D}&timesteps=1d&units=imperial&timezone=America/Los_Angeles&apikey=${app_key}`
    const Tomo_API_1H = `https://api.tomorrow.io/v4/timelines?location=${latitude},${longitude}&fields=${Fields_1H}&timesteps=1h&units=imperial&timezone=America/Los_Angeles&apikey=${app_key}`
    
    const [Response_1D, Response_1H] = await Promise.all([
      fetch(Tomo_API_1D),
      fetch(Tomo_API_1H),
    ]);

    const [Data_1D, Data_1H] = await Promise.all([
      Response_1D.json(),
      Response_1H.json(),
    ]);

    Processed_Data_1D = Data_Process_1D(Data_1D)
    Processed_Data_1H = Data_Process_1H(Data_1H)
    
    return res.status(200).json({
      Data_1D: Processed_Data_1D,
      Data_1H: Processed_Data_1H,
    });

  } catch (error) {
    console.error('Error processing get weather data:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

app.get('/Get_autocomplete', async (req, res) => {
  const { input } = req.query;
  const googleApiKey = 'AIzaSyBOI4I20qLdWyWRU3ha2r_SErsdCqTkCRg'; 
  const url = `https://maps.googleapis.com/maps/api/place/autocomplete/json?input=${input}&types=(cities)&key=${googleApiKey}`;

  try {
    const response = await fetch(url);
    const data = await response.json();

    if (!data) {
      console.log('No data');
      res.status(404).send('No data found');
    } else {
      res.json(data);
    }
  } catch (error) {
    console.error('Error fetching autocomplete data:', error);
    res.status(500).send('Error fetching autocomplete data');
  }
});

// Start the server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server is running on http://0.0.0.0:${PORT}`);
});