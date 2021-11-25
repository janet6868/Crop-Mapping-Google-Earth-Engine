// This example demonstrates the use of the Landsat 8 Collection 2, Level 2
// QA_PIXEL band (CFMask) to mask unwanted pixels.
Map.centerObject(kofa, 14); 
Map.addLayer(kofa, {color: 'green'}, 'image');

//var Kofa = ee.Geometry.Polygon([8.2343635423279018,11.5365866666666683, 8.2905699222882845,11.5780959460610120]);
//var geometry = Kofa.geometry();
function maskL8sr(image) {
  // Bit 0 - Fill
  // Bit 1 - Dilated Cloud
  // Bit 2 - Cirrus
  // Bit 3 - Cloud
  // Bit 4 - Cloud Shadow
  var qaMask = image.select('QA_PIXEL').bitwiseAnd(parseInt('11111', 2)).eq(0);
  var saturationMask = image.select('QA_RADSAT').eq(0);

  // Apply the scaling factors to the appropriate bands.
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);

  // Replace the original bands with the scaled ones and apply the masks.
  return image.addBands(opticalBands, null, true)
      .addBands(thermalBands, null, true)
      .updateMask(qaMask)
      .updateMask(saturationMask);
}

//Region of interest
var roi = kofa;

///%%%%%%%%%%%%%%% CLASSIFICATION  %%%%%%%%%%%%%%%%%%%%%//
   // Load Landsat 5 input imagery.
var landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  // Filter to get only one year of images.
  .filter(ee.Filter.date('2015-05-01', '2015-12-01'))
  // Filter to get only images under the region of interest.
  .filter(ee.Filter.bounds(roi))
  // Sort by scene cloudiness, ascending.
  .sort('CLOUD_COVER')
  .map(maskL8sr)
  .select('SR_B.*');

//the composite image
var composite = landsat.median().clip(roi);


//%%%%%%% Hyperparameter Tuning%%%%%%%%%%%
//(b4 - 2*b1 + b3) / (b4 + 2*b1 + b3)
//((b4 - b1) / (b4 + (c1*b1) - (c2*b3) + l))*2.5
//return (b4 - b3) / (b4 - b1)

var addIndices = function(image) {
var ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename(['ndvi']);
  var ndbi = image.normalizedDifference(['SR_B6', 'SR_B5']).rename(['ndbi']);
  //var mndwi = image.normalizedDifference(['SR_B3', 'SR_B6']).rename(['mndwi']); 
  var bsi = image.expression(
      '(( X + Y ) - (A + B)) /(( X + Y ) + (A + B)) ', {
        'X': image.select('SR_B6'), //swir1
        'Y': image.select('SR_B4'),  //red
        'A': image.select('SR_B5'), // nir
        'B': image.select('SR_B2'), // blue
  }).rename('bsi');
  var arvi = image.expression(
      '(( A-2*Y + C)) /((A+2*Y + C)) ', {
        'X': image.select('SR_B6'), //swir1
        'Y': image.select('SR_B4'),  //red
        'A': image.select('SR_B5'), // nir
        'B': image.select('SR_B2'), // blue
        'C': image.select('SR_B3'), // Green
  }).rename('arvi');
  var evi = image.expression(
      '((( A - Y)) /((A +(0.5*C)-(0.5*B)+1)))*2.5 ', {
        'X': image.select('SR_B6'), //swir1
        'Y': image.select('SR_B4'),  //red
        'A': image.select('SR_B5'), // nir
        'B': image.select('SR_B2'), // blue
        'C': image.select('SR_B3'), // Green
  }).rename('evi');
   var sipi = image.expression(
      '(( A - B)) /((A - Y))', {
        'X': image.select('SR_B6'), //swir1
        'Y': image.select('SR_B4'),  //red
        'A': image.select('SR_B5'), // nir
        'B': image.select('SR_B2'), // blue
        'C': image.select('SR_B3'), // Green
  }).rename('sipi');
  return image.addBands(ndvi).addBands(ndbi).addBands(bsi).addBands(arvi).addBands(evi)
}

var composite = addIndices(composite);

//%%%%%%%%%%% Creating training dataset from the Ground Truth %%%%%%

// Apply filter where grown crop values equals 0-4.
var millet_sorghum = kofa.filter(ee.Filter.eq('Grown_crop', 0));
print("millet_sorghum", millet_sorghum);
var soybean_sorghum = kofa.filter(ee.Filter.eq('Grown_crop', 1));
print( "soybean_sorghum", soybean_sorghum);

var rice_sorghum = kofa.filter(ee.Filter.eq('Grown_crop', 2));
print("rice_sorghum", rice_sorghum);

var groundnut_sorghum = kofa.filter(ee.Filter.eq('Grown_crop', 3));
print("groundnut_sorghum", groundnut_sorghum);

var maize_sorghum = kofa.filter(ee.Filter.eq('Grown_crop', 4));
print( "maize_sorghum",maize_sorghum);

//Merge the crops to be sued for training
var crops_collection = millet_sorghum.merge(soybean_sorghum).merge(rice_sorghum).merge(groundnut_sorghum).merge(maize_sorghum);
print('crops_collection',crops_collection);

 //Add a random column and split the GCPs into training and validation sevar 
 // Use these bands for classification.
var bands = ['SR_B2', 'SR_B3', 'SR_B4','SR_B5','SR_B6','SR_B7'];
// The name of the property on the points storing the class label.
var classProperty = 'Grown_crop';

// Sample the composite to generate training data.  Note that the
// class label is stored in the 'landcover' property.
var training = composite.select(bands).sampleRegions({
  collection: crops_collection,
  properties: [classProperty],
  scale: 30
});

print('training',training);

//%%%%%% Unsupervised classification %%%%%%%%%%%%%%%%
// Instantiate the clusterer and train it.
var clusterer = ee.Clusterer.wekaKMeans(5).train(training);

// Cluster the input using the trained clusterer.
var unsupervised_classification = composite.cluster(clusterer);

// Display the clusters with random colors.
Map.addLayer(unsupervised_classification.randomVisualizer(), {}, 'clusters');
//

//%%%%%%%%%% Supervised classification%%%%%%%%%%%%

// Train a CART classifier.
var classifier = ee.Classifier.smileCart().train({
  features: training,
  classProperty: classProperty,
});
// Print some info about the classifier (specific to CART).
print('CART, explained', classifier.explain());

// Classify the composite.
var classified = composite.classify(classifier);

//Map.centerObject(crops_collection);
Map.addLayer(classified, {min: 0, max: 4, palette: ['red', 'green', 'blue','yellow','darkorange']},'cart_classified');

// Optionally, do some accuracy assessment.  Fist, add a column of
// random uniforms to the training dataset.
var withRandom = training.randomColumn('random');

// We want to reserve some of the data for testing, to avoid overfitting the model.
var split = 0.7;  // Roughly 70% training, 30% testing.
var trainingPartition = withRandom.filter(ee.Filter.lt('random', split));
var testingPartition = withRandom.filter(ee.Filter.gte('random', split));


print('training_partition', trainingPartition);

// Display training and validation points to see distribution within the AOI.


// Trained with 70% of our data.
var trainedClassifier = ee.Classifier.smileRandomForest(200).train({
  features: trainingPartition,
  classProperty: classProperty,
  inputProperties: bands
});


// Classify the test FeatureCollection.
var test = testingPartition.classify(trainedClassifier);

// Print the confusion matrix.
var RF_confusionMatrix = test.errorMatrix(classProperty, 'classification');
print('RF_Confusion_Matrix', RF_confusionMatrix);

// Printing of confusion matrix may time out. Alternatively, you can export it as CSV

print('RF_test accuracy', RF_confusionMatrix.accuracy());
//Map.center(crops_collection);
Map.addLayer(classified, {min: 0, max: 4, palette: ['red', 'green', 'blue','yellow','darkorange']},'RF_classified');

// Tune the numberOfTrees parameter.
var numTreesList = ee.List.sequence(10, 150, 10);

var accuracies = numTreesList.map(function(numTrees) {
  var classifier = ee.Classifier.smileRandomForest(numTrees)
      .train({
        features: trainingPartition,
        classProperty: classProperty,
        inputProperties: bands
      });

  // Here we are classifying a table instead of an image
  // Classifiers work on both images and tables
  return test
    .classify(classifier)
    .errorMatrix('Grown_crop', 'classification')
    .accuracy();
});


var chart = ui.Chart.array.values({
  array: ee.Array(accuracies),
  axis: 0,
  xLabels: numTreesList
  }).setOptions({
      title: 'Hyperparameter Tuning for the numberOfTrees Parameters',
      vAxis: {title: 'Validation Accuracy'},
      hAxis: {title: 'Number of Tress', gridlines: {count: 15}}
  });
print(chart);

//************************************************************************** 
// Exporting Results
//************************************************************************** 

// Create a Feature with null geometry and the value we want to export.
// Use .array() to convert Confusion Matrix to an Array so it can be
// exported in a CSV file
var feature_confM = ee.FeatureCollection([
  ee.Feature(null, {
    'accuracy': RF_confusionMatrix.accuracy(),
    'matrix': RF_confusionMatrix.array()
  })
  ]);
print('Rf_confusion matrix details',feature_confM);

Export.table.toDrive({
  collection: feature_confM,
  description: 'Accuracy_Export',
  folder: 'Earthengine',
  fileNamePrefix: 'accuracy',
  fileFormat: 'CSV'
});


// Export the image, specifying scale and region.
Export.image.toDrive({
  image: classified,
  description: 'RF_classification',
  scale: 10,
  maxPixels: 1e13,
  region: kofa
});

// Export the image, specifying scale and region.
Export.image.toDrive({
  image: kofa,
  description: 'input_image',
  scale: 10,
  maxPixels: 1e13,
  region:kofa
});


// Create an SVM classifier with custom parameters.
var classifier = ee.Classifier.libsvm({
  kernelType: 'RBF',
  gamma: 1,
  cost: 10
});

// Train the classifier.
var trained = classifier.train(trainingPartition, classProperty, bands);

// Classify the image.
var svm_classified = testingPartition.classify(trained);

// Print the confusion matrix.
var svm_confusionMatrix = svm_classified.errorMatrix(classProperty, 'classification');
print('svm_confusionMatrix', svm_confusionMatrix);

// Printing of confusion matrix may time out. Alternatively, you can export it as CSV

print('svm_test accuracy', svm_confusionMatrix.accuracy());
// Display the classification result and the input image.

Map.addLayer(svm_classified,
             {min: 0, max: 4, palette: ['red', 'green', 'blue','yellow','darkorange']}, 'SVM_classified');
             
// Create the panel for the legend items.
var legend = ui.Panel({
  style: {
    position: 'middle-left',
    padding: '15px 15px'
  }
});           
             
var legendTitle = ui.Label({
  value: 'Kofa, Nigeria Land Cover',
  style: {
    fontWeight: 'bold',
    fontSize: '18px',
    margin: '0 0 7px 0',
    padding: '1'
  }
});
legend.add(legendTitle);

var loading = ui.Label('Loading legend...', {margin: '4px 0 4px 0'});
legend.add(loading);

// Creates and styles 1 row of the legend.
var makeRow = function(color, name) {
      
      // Create the label that is actually the colored box.
      var colorBox = ui.Label({
        style: {
          backgroundColor: color,
          // Use padding to give the box height and width.
          padding: '8px',
          margin: '1 0 4px 0'
        }
      });
      
      // Create the label filled with the description text.
      var description = ui.Label({
        value: name,
        style: {margin: '0 0 4px 6px'}
      });
      
      // return the panel
      return ui.Panel({
        widgets: [colorBox, description],
        layout: ui.Panel.Layout.Flow('horizontal')
      });
};


//  Palette with the colors
var palette =['red', 'green', 'blue','yellow','darkorange'];

// name of the legend
var names = ["millet_sorghum","soybean_sorghum","rice_sorghum","groundnut_sorghum","maize_sorghum"];

for (var i = 0; i < names.length; i++) {
    legend.add(makeRow(palette[i], names[i]));
  }
Map.addLayer(svm_classified,
             {min: 0, max: 4, palette: palette},
             'SVM_classified');


// Add the legend to the map.

Map.add(legend);

