const mongoose = require("mongoose");

module.exports = (res, callback) => {
  mongoose.connect("mongodb://localhost/unstableunicorns", {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  });

  const db = mongoose.connection;
  db.on("error", console.error.bind(console, "connection error:"));

  db.once("open", async function () {
    const query = await callback();

    db.close();
    return res ? res.json(query) : query;
  });
};
