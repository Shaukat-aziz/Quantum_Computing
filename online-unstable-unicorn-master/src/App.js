import React from "react";
import "./App.css";
import store from "./store.js";
import { Provider } from "react-redux";
import { BrowserRouter as Router, Route } from "react-router-dom";
import HomePage from "./pages/Home/HomePage.js";
import LobbyPage from "./pages/Lobby/LobbyPage.js";
import MemoGamePage from "./pages/Game/GamePage.js";

console.log(store.getState());
store.subscribe(() => console.log(store.getState()));

function App() {
  return (
    <Provider store={store}>
      <Router>
        <div className="App">
          <Route exact path="/app">
            <HomePage />
          </Route>

          <Route path="/app/:id/lobby">
            <LobbyPage />
          </Route>

          <Route path="/app/:id/game">
            <MemoGamePage />
          </Route>
        </div>
      </Router>
    </Provider>
  );
}

export default App;
