// #############################################################################
// ### Data (sentinel 2)  Preparation and feature extraction###
// #############################################################################

print(studysite);
Map.centerObject(studysite, 10);  

//Function to mask clouds S2
function maskS2srClouds(data) {
  var qa = data.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return data.updateMask(mask).divide(10000);
}

// Filter Sentinel-2 collection
var input = sent2.filterBounds(studysite)
                .filterDate("2019-06-06", "2019-07-31")
                .filterMetadata('CLOUDY_PIXEL_PERCENTAGE',
                                'less_than',50)
                .map(maskS2srClouds);
print(input);    //image with the unmasked pixels          


//Exploring image collection and its metadata
print("A Sentinel-2 scene:", input);

 //Get the number of images by converting the collection into a list
var size = input.toList(100).length();
print('The number of images:', size);

/*//Get statistics for a property of the images in the collection
//See how structure varies, though not important in this case
var sunstats = input.aggregate_stats('SUN_ELEVATION');
print('Thr sun elevation stats:', sunstats);*/

//sort by cloud cover property
var imagecloud = ee.Image(input.sort('CLOUD_COVER').first());
print('Least cloudy image:', imagecloud);


//Creating a mosaic and composite from mosaic
//var mosaic =ee.ImageCollection([image1,image2]);
//var composite =mosaic.max();

// Composite images
var s2composite = input.median().clip(studysite); // can be changed to mean, min, etc 

// Add composite to map
Map.addLayer(s2composite,{bands:['B4','B3','B2'],min:0.02,max:0.3,
                          gamma:1.5},'Sentinel-2 2019 composite', false);


// Export the composite image to your Google Drive
Export.image.toDrive({
  image: s2composite,
  description: 'composite2019', // task name to be shown in the Tasks tab
  scale: 10, // the spatial resolution of the image
  region: studysite,
  maxPixels: 1e13    //maximum number of pixels
});


/*###################################################
 ### Image projection ###
 ########################################*/
//var imageProj = ee.Image("COPERNICUS/S2/20190706T110656_20190801T015159_T30SVG");
var imageProj =imagecloud.select('B4');
//first.select['B4']
//print the projection
print('Projection and transformation information:',imageProj.projection());

//print the nominal pixel size of the image in meters at the lowestlevel of image pyramid
print('Pixel size in meters:',imageProj.projection().nominalScale());

/* #############################################################################
### Vegetation Indices calculation ###
#############################################################################*/
//NDVI Calculation
var image1 = ee.Image(sent2
             .filterDate("2019-06-06", "2019-07-31")
             .filterBounds(studysite)
             .sort("CLOUD_COVER")
             .first())
             .clip(studysite);
             
var image2 = ee.Image(sent2
             .filterDate("2019-08-11", "2019-08-31")
             .filterBounds(studysite)
             .sort("CLOUD_COVER")
             .first())
             .clip(studysite);


//create a function that computes the NDVI
var getNDVI = function(image){
  return image.normalizedDifference(['B8','B4']);
};
//Display the individual scenes
Map.addLayer(image1,{bands:["B4", "B3", "B2"]}, 'Ist image');
Map.addLayer(image2, {bands:["B4", "B3", "B2"]}, '2nd image');

//Compute NDVI from the scenes
var ndvi1 = getNDVI(image1);
var ndvi2 = getNDVI(image2);

//Display ndvi for each image
Map.addLayer(ndvi1,{},'1st image NDVI');
Map.addLayer(ndvi2,{},'2nd image NDVI');

//calculate the difference in NDVI for the two images
var ndviDifference = ndvi2.subtract(ndvi1);
Map.addLayer(ndviDifference,{},'Difference NDVI', false);

// Create color ramp based on bounds and colors you're interested in
var NDVI_ramp =
  '<RasterSymbolizer>' +
    '<ColorMap type="ramp" extended="false" >' +
      '<ColorMapEntry color="#0000ff" quantity="0" label="0"/>' +
      '<ColorMapEntry color="#00ff00" quantity="0.3" label="0.3" />' +
      '<ColorMapEntry color="#007f30" quantity="0.5" label="0.5" />' +
      '<ColorMapEntry color="#30b855" quantity="0.7" label="0.7" />' +
    '</ColorMap>' +
  '</RasterSymbolizer>';

//Display the RGB image 
Map.addLayer(input, {bands: ["B4", "B3", "B2"]}, 'input_image', false);

// Apply sldStyle color ramp to your ndvi image and display it.
Map.addLayer(ndviDifference.sldStyle(NDVI_ramp), {}, "NDVI with ramp", false);

// Reduce the image collection/region using the `autoHistogram`
var hist = ndviDifference.reduceRegion({
  reducer: ee.Reducer.autoHistogram(),
  geometry: studysite,
  scale: 1000,
  bestEffort: true,
});

print('The reduced pixels values:',hist);
// The result of the region reduction by `autoHistogram` is an array. Get the
// array and cast it as such for good measure.
var histArray = ee.Array(hist.get('nd'));

// Subset the values that represent the bottom of the bins and project to
// a single dimension. Result is a 1-D array.
var binBottom = histArray.slice(1, 0, 1).project([0]);

// Subset the values that represent the number of pixels per bin and project to
// a single dimension. Result is a 1-D array.

var nPixels = histArray.slice(1, 1, null).project([0]);

// Chart the two arrays using the `ui.Chart.array.values` function.
var histColumnFromArray = ui.Chart.array.values({
  array:nPixels,
  axis: 0,
  xLabels: binBottom})
  .setChartType('ColumnChart');
print(histColumnFromArray);

/*Histogram chart using a feature collection*/
// Cast the histogram table array as a list.
var histList = histArray.toList();

// Map over the list to create a list of features per bin and set respective
// bin bottom and number of pixels properties.
var featureList = histList.map(function(bin) {
  bin = ee.List(bin);
  var props = {
    binBottom: ee.Number(bin.get(0)),
    nPixels: ee.Number(bin.get(1))
  };
  return ee.Feature(studysite, props);
});

// Convert the feature list to a feature collection.
var featureCol = ee.FeatureCollection(featureList);

Export.table.toAsset({
  collection: featureCol,
  description: 'histogram_table',
  assetId: 'histogram_table'
});

// Chart histogram from the constructed feature collection as a line chart.
var histLineFromFc = ui.Chart.feature.byFeature({
  features: featureCol,
  xProperty: 'binBottom',
  yProperties: ['nPixels']})
  .setChartType('LineChart');
print(histLineFromFc);

// Chart histogram from the constructed feature collection as a column chart.
var histColumnFromFc = ui.Chart.feature.byFeature({
  features: featureCol,
  xProperty: 'binBottom',
  yProperties: ['nPixels']})
  .setChartType('ColumnChart');
  print(histColumnFromFc); 


// Prints a histogram of the polygons
var print_hist = function(input, poly, title){
  var options = {
    title: 'Histogram',
    fontSize: 20,
    hAxis: {title: 'DN'},
    vAxis: {title: 'count of DN'},
  };
  
  var histogram = ui.Chart.image.histogram(hist, studysite, 30)
                      .setOptions(options);
      print(histogram);
};

// Export the image, specifying scale and region.
Export.image.toDrive({
  image: ndviDifference,
  description: 'ndvi_ramp',
  scale: 30,
  region: studysite,
  maxPixels: 1e13
});
 
/*//making the greenest pixel composite
var withNDVI = sent2.add(getNDVI);

// Make a "greenest" pixel composite.
var greenest = withNDVI.qualityMosaic('ndvi1');

// Display the result.
var visParams = {bands: ['B4', 'B3', 'B2'], max: 0.3};
Map.addLayer(greenest, visParams, 'Greenest pixel composite');*/

/*###############################################################################
 ### Classification ###
 #############################################################################*/
//Instantiate the unsupervised classification
//sample the image to construct 'training sample of pixels to charactize
var unsupervised_input = sent2.filterBounds(studysite)
                .filterDate("2019-06-06", "2019-07-31")
                .sort('CLOUD_COVER')
                .mosaic()
                .clip(studysite);
                
//set map display to centre on the study site             
Map.centerObject(studysite,10);

//Display the input image as a cir composite
Map.addLayer(unsupervised_input,{bands: ['B4','B3','B2']},'unsupervised_input_image');

//sample the image to construct a training sample of pixels to characterise the image.
var training = unsupervised_input.sample({ 
  region: studysite,
  scale:10,
  numPixels: 5000
});

//use wekameans for classification
var clusterer = ee. Clusterer.wekaKMeans(7).train(training);

//results
var results = unsupervised_input.cluster(clusterer);
print(results);

//Display the classified image with random colours of each classes
Map.addLayer(results.randomVisualizer(), {},'unsupervised_classified_image',false);

            /*############# Supervised classification ##########################*/
            
//select the first image; the scene with the least clouds in the sorted collection
var supervised_input_image = ee.Image(input.first()) //the first scene
                    .clip(studysite);//clip the selcted image to the studysite

//Set the display to the center on the study area
Map.centerObject(studysite, 10); 

 //Display the input image as a color infrared composite
Map.addLayer(supervised_input_image,{bands: ['B4','B3','B2']},'super_input_image', false);

/*Combine the manually trained data of the crops into a reference dataset*/
var crops_polygons = Maize.merge(Cassava).merge(Bean).merge(Maize_Bean).merge(Maize_cassava).merge(Sorghum_soybean).merge(Cassava_Bean);
print(crops_polygons);

/*use the crop polygons data to sample the classified image and create a new band called landcover
that containds the correpsonding value for each refernce point based on the classified image.*/
var trainingPixels= supervised_input_image.sampleRegions({
  collection: crops_polygons,
  properties: ['landcover'],
  scale: 10
});

// Select the bands for training
var bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'];


// Splits data into training and validation groups. Set at 70% training/30% validation
var splitData = function(data){
  var dict = {};
  var randomTpixels = data.randomColumn(); 
  var trainingData = randomTpixels.filter(ee.Filter.lt('random', 0.7));
  var valiData = randomTpixels.filter(ee.Filter.gte('random', 0.7));
  
  dict.training = trainingData;
  dict.validation = valiData;
  
  return dict;
  
};

var cropdata = splitData(trainingPixels); 

var cropclassifiedTD = ee.Classifier.smileRandomForest(100).train(cropdata.training, "landcover", bands);


// Creates error matrix
var createMatrix = function(data){
  var trainAccuracy = data.errorMatrix("landcover", "classification");
  print('RF Resubstitution error matrix: ', trainAccuracy);
  print('RF Training overall accuracy: ', trainAccuracy.accuracy());
};

var cropvalidation =cropdata.validation.classify(cropclassifiedTD); // Classifies the validation data

createMatrix(cropvalidation); // Print the error matrix

/*var Rfcoefficients = ee.Array([[56,63,0,2,35,0,0],
  [2,22556,13,39,170,2,83],
  [0,112,57,0,5,0,14],
  [0,1817,0,352,6,0,89],
  [6,846,0,9,1785,5,101],
  [0,64,1,1,31,66,20],
  [0,176,1,11,20,0,1274]]);

print(Rfcoefficients.length());*/
var classified = supervised_input_image.classify(cropclassifiedTD);

var palette = [
  'c57f1f', //maize (0) //brown
  '999900', //Cassava(1) //red
  '0b4a8b', // Common bean (2) //blue
  'eb0dff',// maize_common bean (3) //purple
  'ff830e',// maize_cassava (4) // orange
  'ecf430', //maize_soybean (5) //yellow
  '1cff1e' //cassava_bean (6)//green
];

Map.addLayer(classified,
            {palette: palette, min: 0, max: 3}, // min and max indicate that there will be 4 classes coloured
            "RFsupervised_classified_field", false); // Add the classification to the map with the four classes being coloured according to ABcolours


// Export the image, specifying scale and region.
Export.image.toDrive({
  image: classified,
  description: 'westk_supervised_classification',
  scale: 30,
  region: studysite
});

// Export the image, specifying scale and region.
Export.image.toDrive({
  image: supervised_input_image,
  description: 'input_image',
  scale: 30,
  region: studysite,
  maxPixels: 1e13
});
// #############################################################################
// ### Class Area calculation ###  this is important for understanding the range of the crops--for
//intensity purposes
// #############################################################################
var classArea = function(classified){
  var areaImage = ee.Image.pixelArea().addBands(
        classified);
   
  var areas = areaImage.reduceRegion({ 
        reducer: ee.Reducer.sum().group({
        groupField: 1,
        groupName: 'classification',
      }),
      geometry: classified.geometry(),
      scale: 30, 
      maxPixels: 1e8
      }); 
  
  var classAreas = ee.List(areas.get('groups'));
   
  var classAreaLists = classAreas.map(function(item) { // Function within a function to create a dictionary with the values for every group
    var areaDict = ee.Dictionary(item);
    var classNumber = ee.Number(areaDict.get('classification')).format();
    var area = ee.Number(
      areaDict.get('sum')).divide(1e6).round(); // The result will be in square meters, this converts them into square kilometers
    return ee.List([classNumber, area]);
  });
   
  var result = ee.Dictionary(classAreaLists.flatten()); // Flattens said dictionary so it is readable for us
  return(result);
};
//Values in kms for the area of each class:
print('validated:', classArea(classified));

//creating a chart of the classes

var create_chart = function(classification, AOI, classList){ // for classList, create a list of your classes as strings
  var options = {
    hAxis: {title: 'Class'},
    vAxis: {title: 'Area'},
    title: 'Area by class',
    series: { // You can change these to be whatever colours you'd like. Simply add numbers to match how many classes you have
      0: {color: 'brown'},
      1: {color: 'red'},
      2: {color: 'blue'},
      3: {color: 'purple'},
      4: {color: 'orange'},
      5: {color: 'yellow'},
      6: {color: 'green'}}
  }; 
  var areaChart = ui.Chart.image.byClass({
    image: ee.Image.pixelArea().addBands(classification),
    classBand: 'classification', 
    scale: 30,
    region: AOI,
    reducer: ee.Reducer.sum()

  }).setSeriesNames(classList)
  .setOptions(options)
  ;
  print(areaChart);
};


/*################# CART Classification##################*/

// Train a CART classifier with default parameters.
var cart_trained = ee.Classifier.smileCart().train(cropdata.training, "landcover", bands);

// Classify the image with the same bands used for training.
var cart_classified = supervised_input_image.select(bands).classify(cart_trained);

print('Cart_classified_image',cart_classified);

// Creates error matrix
var createMatrix2 = function(data){
  var trainAccuracy = data.errorMatrix("landcover", "classification");
  print('cart Resubstitution error matrix: ', trainAccuracy);
  print('cart Training overall accuracy: ', trainAccuracy.accuracy());
};

var cropvalidation2 =cropdata.validation.classify(cart_trained); // Classifies the validation data

createMatrix2(cropvalidation2); // Print the error matrix

/*
var cartcoeffiecients = ee.Array([[75,55,0,1,21,1,3],
  [87,20349,86,1357,730,62,194],
  [0,65,96,11,10,0,6],
  [5,1231,9,924,22,1,72],
  [22,645,17,39,1902,35,92],
  [1,50,1,4,26,85,16],
  [1,158,18,49,78,17,1161]]);    
print(cartcoefficients.length());*/
var palette = [
  'c57f1f', //maize (0) //brown
  '999900', //Cassava(1) //red
  '0b4a8b', // Common bean (2) //blue
  'eb0dff',// maize_common bean (3) //purple
  'ff830e',// maize_cassava (4) // orange
  'ecf430', //maize_soybean (5) //yellow
  '1cff1e' //cassava_bean (6)//green
];

// Display the inputs and the results.
Map.centerObject(studysite, 10);
//Map.addLayer(supervised_input_image, {bands: ['B4', 'B3', 'B2'], max: 0.4}, 'image');
Map.addLayer(cart_classified,
             {min: 0, max: 2, palette: ['red', 'green', 'blue']},
             'cart_classification',false);
             
/*################# SVM Classification##################*/             
// Create an SVM classifier with custom parameters.
var classifier = ee.Classifier.libsvm({
  kernelType: 'RBF',
  gamma: 0.5,
  cost: 10
});

// Train the classifier.
var svmTrained = classifier.train(cropdata.training, "landcover", bands);

// Creates error matrix
var createMatrix = function(data){
  var trainAccuracy = data.errorMatrix("landcover", "classification");
  print('Svm Resubstitution error matrix: ', trainAccuracy);
  print('Svm Training overall accuracy: ', trainAccuracy.accuracy());
};

var cropvalidation3 =cropdata.validation.classify(svmTrained); // Classifies the validation data

createMatrix(cropvalidation3); // Print the error matrix
//var ceo = // Get the 1x6 greenness slice, display it.

/*var svmcoefficients = ee.Array([
                            [0, 152, 0, 0, 0,0, 4],
                           [ 0, 22687,0,, 0,0,  178],
                            [0,163,  0,0,0, 0, 25 ],
                            [0,22,0,0,0,0,242],
                            [0,2405,0,1,346],
                            [0, 145, 0, 0, 0, 0, 38],
                            [0, 452,0, 0, 0, 0, 1030]]);
// Print the dimensions.
print(svmcoefficients.length()); //    [6,6]*/
var palette = [
  'c57f1f', //maize (0) //brown
  '999900', //Cassava(1) //red
  '0b4a8b', // Common bean (2) //blue
  'eb0dff',// maize_common bean (3) //purple
  'ff830e',// maize_cassava (4) // orange
  'ecf430', //maize_soybean (5) //yellow
  '1cff1e' //cassava_bean (6)//green
];

// Classify the image.
var svmClassified = supervised_input_image.classify(svmTrained);

//print(svmClassified);
//palette
var palette = [
  'c57f1f', //maize (0) //brown
  '999900', //Cassava(1) //red
  '0b4a8b', // Common bean (2) //blue
  'eb0dff',// maize_common bean (3) //purple
  'ff830e',// maize_cassava (4) // orange
  'ecf430', //maize_soybean (5) //yellow
  '1cff1e' //cassava_bean (6)//green
];
// Display the classification result and the input image.
Map.setCenter(34.3619, -0.482399, 10);
//Map.addLayer(supervised_input_image, {bands: ['B4', 'B3', 'B2'], max: 0.5, gamma: 2});
Map.addLayer(studysite, {}, 'training polygons');
Map.addLayer(svmClassified,
             {min: 0, max: 1, palette:palette},
             'Svm classified');
             
// Get the 1x6 greenness slice, display it.



