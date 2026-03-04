// firebase.js
import { initializeApp } from "firebase/app";
import { getDatabase } from "firebase/database";

const firebaseConfig = {
  apiKey: "AIzaSyD3ibELSA7pjag84KA0tliK0ehFoHeOfh0",
  authDomain: "crop-recommendation-3f8a5.firebaseapp.com",
  databaseURL: "https://crop-recommendation-3f8a5-default-rtdb.firebaseio.com",
  projectId: "crop-recommendation-3f8a5",
  storageBucket: "crop-recommendation-3f8a5.firebasestorage.app",
  messagingSenderId: "900862709245",
  appId: "1:900862709245:web:2079530e434d6a756edd94",
  measurementId: "G-D5FSK9XV4W"
};

const app = initializeApp(firebaseConfig);

export const database = getDatabase(app); 