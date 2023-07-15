import { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import golden from './golden.gif'
//import {motion} from 'framer-motion';

const App = () => {

  const [image, setImage] = useState(null);
  const[result, setResult] = useState(null);
  const [resultUrl, setResultUrl] = useState('');

  //updates image 
  const updateImage = (event) => {
    setImage(event.target.files[0]);
    console.log("successful update");
  }

  //to da back
  const handleUpload = (event) => {
    event.preventDefault();
    if (image) {
      const formData = new FormData();
      formData.append('image', image);
      try {
          axios.post('/predict', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',},})
            .then(response => 
              { 
                setResult(response.data.prediction);
                setResultUrl(response.data.doodle_url)
              });
          console.log('successfully sent');
          
      } catch (error) {
        console.error('failed :(', error.response.data);
      }
    }
  };

  return (
    <main className="app">
      <br />
      <div class="header">
        <h1>get a doodle of your doggie!</h1>
        <p>upload a picture of your pup and receive a customized doodle!<br/> make sure it's mostly of their head for best results.  </p>
        <img src={golden} alt="golden"></img>
      </div>
      <div class="container">
        { /*<p>**DISCLAIMER!! my neural network is only trained on some breeds. your pup might not get an accurate doodle, <br/> because my computer could not handle testing more. but im sure your dog is beautiful!! </p>*/ }
        <form onSubmit={handleUpload}>
        <label for="file-upload" class="custom-file-upload">
            upload dog pic
        </label>
          <input id="file-upload"
            type="file"
            name="image"
            accept="image/jpg, image/jpeg, image/png"
            onChange={updateImage}>
            </input>
            <button class="button" type="submit">get my doodle</button>
          </form>
      </div>
      <div class="uploaded">
          {image && <p>image successfully uploaded!</p>}
      </div>
      <div class="doodle">
          {result && <p>here is your pup doodle!!</p>}
          {resultUrl && <img src={resultUrl} alt="breed"></img>}
          {resultUrl && <button class="reset" onClick ={ () => {
            setImage(null);
            setResult(null);
            setResultUrl('');
          }} 
          >try again</button>}
        </div>
    </main>
  );
}

export default App;