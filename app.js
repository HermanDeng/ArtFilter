var express = require("express"),
	app = express(),
	fileUpload = require("express-fileupload"),
	bodyParser = require("body-parser");



app.use(bodyParser.urlencoded({extended:true}));
app.set("view engine", "ejs");
app.use(express.static(__dirname + "/public"));
app.use(fileUpload());


//Routers
app.get("/", function(req, res){
	res.render('homepage');
});


app.get("/portfolio", function(req, res){
	res.render("portfolio");
});


app.get("/portfolio/new", function(req, res){
	res.render("new");
});

app.get("/uploaded", function(req, res){
	res.render("uploaded");
});


app.post("/portfolio", function(req, res){
	if (!req.files) {
		return res.status(400).send('No files were uploaded');
	}
	let file = req.files.uploadFile;
	file.mv("public/input/input.jpg", function(err){
		if (err) {
			return res.status(500).send(err);
		}
		res.redirect("/portfolio/new");
	});
});


// Did not have time to couple the Python Script to JS 
// My idea is to call Python script to generated the new img.
// now I just did that offline.

app.post("/portfolio/new", function(req, res){
	var style = req.body.style;
	res.redirect("/uploaded");
});



app.listen(3000, "localhost", function(){
	console.log("Please visit: localhost/3000");
});