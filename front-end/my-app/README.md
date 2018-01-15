# front end

1. Run the CUDA executable to generate `list.json` file. It is generated in the `images` folder.
2. Run `npm install http-server -g` to install node `http-server`
3. Run `http-server ./ --cors` in the same directory as `list.json` to host a static server with CORS enabled.
3. Run `npm install` and `npm start` to start the server.