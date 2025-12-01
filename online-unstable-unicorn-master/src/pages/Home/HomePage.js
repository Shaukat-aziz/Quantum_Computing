import "./HomePage.css";
import React, { useState, useEffect, useRef } from "react";
import { useSelector, useDispatch } from "react-redux";
import { joinLobby, startGame } from "actions";
import { useHistory } from "react-router-dom";
import {
  Header,
  Segment,
  Button,
  Divider,
  Input,
  Card,
  Icon,
} from "semantic-ui-react";
const colors = ["purple", "blue", "teal", "green", "yellow", "orange", "red"];

function HomePage() {
  const inputEl = useRef(null);
  const [newLobbyName, setNewLobbyName] = useState("");
  const [lobbies, setLobbies] = useState([]);
  const history = useHistory();
  const socketServer = useSelector((state) => state.socket);
  const dispatch = useDispatch();
  const setLobby = (lobby) => dispatch(joinLobby(lobby));

  useEffect(() => {
    inputEl.current.focus();
    socketServer.emit("getLobbies");
    socketServer.on("returnLobbies", (lobbies) => {
      console.log(lobbies);
      setLobbies(lobbies);
    });
    socketServer.on("lobbyCreated", (lobbies) => {
      setLobbies(lobbies);
    });

    // JUST FOR TESTING
    socketServer.on("startingGame", (game, decks, players) => {
      dispatch(
        startGame(
          game,
          decks,
          players,
          game.turn % players.length ===
            parseInt(localStorage.getItem("currentPlayerIndex"))
        )
      );
      history.push(`/app/${game.uri}/game`);
      socketServer.removeListener("startGame");
    });

    return () => {
      socketServer.removeListener("lobbyCreated");
      socketServer.removeListener("returnLobbies");
      socketServer.removeListener("startGame");
    };
  }, [dispatch, history, socketServer]);

  function joinPublicLobby(lobby) {
    setLobby(lobby);
    socketServer.emit("joinLobby", lobby.name);
    history.push(`/app/${lobby.name}/lobby`);
  }

  function createLobby() {
    const uri = newLobbyName.split(" ").join("");
    setLobby({
      key: `game:${newLobbyName}`,
      name: uri,
    });
    socketServer.emit("createLobby", newLobbyName);
    history.push(`/app/${uri}/lobby`);
  }

  return (
    <Segment id="home-page" basic textAlign="center">
      <Header content="Unstable Unicorns" />
      <Input
        onChange={(e) => {
          setNewLobbyName(e.target.value);
        }}
        value={newLobbyName}
        ref={inputEl}
        action={
          <Button
            color="blue"
            content="Create Lobbby"
            icon="add"
            onClick={createLobby}
          />
        }
        icon="search"
        iconPosition="left"
        placeholder="lobby name"
      />

      <Divider horizontal>Or Join</Divider>

      <Card.Group>
        {lobbies.map((lobby, index) => {
          const color = colors[index % colors.length];

          return (
            <Card link color={color} key={lobby.name}>
              <Card.Content header={lobby.name} />
              <Card.Content extra>
                <Icon name="user" />
                {lobby.players} Unicorns
              </Card.Content>
              <Card.Content extra>
                <Button
                  fluid
                  onClick={() => {
                    joinPublicLobby(lobby);
                  }}
                  color={color}
                  content="Join Game"
                  icon="add"
                  labelPosition="left"
                />
              </Card.Content>
            </Card>
          );
        })}
      </Card.Group>
    </Segment>
  );
}

export default HomePage;
