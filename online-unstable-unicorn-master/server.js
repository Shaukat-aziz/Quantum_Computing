const express = require("express");
const app = express();
const path = require("path");
const bodyParser = require("body-parser");
const cors = require("cors");
const apiHandler = require("./api");
const socketsHandler = require("./sockets");
const server = app.listen(process.env.PORT || 4404);

// App server configurations
app.use("/app", express.static(path.join(__dirname, "build")));
app.use(cors());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

// Handles all API calls
apiHandler(app);

// Handles all sockets functionality
socketsHandler(server);

// Server routes
app.get("/app/ping", function (req, res) {
  return res.send("pong");
});

app.get("/app", function (req, res) {
  res.sendFile(path.join(__dirname, "build", "index.html"));
});
