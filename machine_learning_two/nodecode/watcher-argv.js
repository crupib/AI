fs = require('fs');
filename = process.argv[2];
if (!filename) {
	throw Error("A file to watch must be specified or exist!");
}
fs.watch(filename,function()
{
	console.log("File "+filename+ "just changed!");
});
console.log("Now watching "+filename+" for changes...");
