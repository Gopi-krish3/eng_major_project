--
-- Dumping data for table `remote_user_clientregister_model`
--

INSERT INTO `remote_user_clientregister_model` (`id`, `username`, `email`, `password`, `phoneno`, `country`, `state`, `city`) VALUES
(1, 'Harish', 'Harish123@gmail.com', 'Harish', '9535866270', 'India', 'Karnataka', 'Bangalore'),
(2, 'Manjunath', 'tmksmanju14@gmail.com', 'Manjunath', '9535866270', 'India', 'Karnataka', 'Bangalore');

-- --------------------------------------------------------

--
-- Dumping data for table `remote_user_detection_accuracy`
--

INSERT INTO `remote_user_detection_accuracy` (`id`, `names`, `ratio`) VALUES
(24, 'Convolutional Neural Network (CNN)', '47.27954971857411'),
(25, 'SVM', '48.592870544090054'),
(26, 'KNeighborsClassifier', '52.72045028142589'),
(27, 'Gradient Boosting Classifier', '50.65666041275797'),
(28, 'Convolutional Neural Network (CNN)', '50.65666041275797');

-- --------------------------------------------------------

--
-- Dumping data for table `remote_user_detection_ratio`
--

INSERT INTO `remote_user_detection_ratio` (`id`, `names`, `ratio`) VALUES
(9, 'Normal Fire', '100.0');

-- --------------------------------------------------------

--
-- Dumping data for table `remote_user_wildfire_danger_forecasting`
--

INSERT INTO `remote_user_wildfire_danger_forecasting` (`id`, `UniqueId`, `AdminUnit`, `CalFireIncident`, `CanonicalUrl`, `Counties`, `CrewsInvolved`, `Dozers`, `Engines`, `Extinguished`, `Fatalities`, `Featured`, `Final`, `Helicopters`, `Injuries`, `Latitude`, `Location`, `Longitude`, `MajorIncident`, `Name`, `PercentContained`, `PersonnelInvolved`, `Description`, `Started`, `Status`, `Updated`, `Prediction`) VALUES
(1, '151.101.1.140-10.42.0.151-443-40597-6', 'CAL FIRE Riverside Unit / San Bernardino National Forest', 'TRUE', '/incidents/2013/8/7/silver-fire/', 'Riverside', '63', '20', '201', '2013-08-12T18:00:00Z', '1', 'FALSE', 'TRUE', '20', '26', '33.86157', 'Poppet Flats Rd near Hwy 243, south of Banning', '-116.90427', 'TRUE', 'Silver Fire', '100', '2106', 'The Silver Fire burned in August 2013 off Poppet Flats Rd near Hwy 243, south of Banning in Riverside County. \r\n', '2013-08-07T14:05:00Z', 'Finalized', '2013-08-12T18:00:00Z', 'Normal Fire'),
(2, '10.42.0.151-52.88.110.117-41227-443-6', 'CAL FIRE San Bernardino Unit', 'TRUE', '/incidents/2013/2/24/river-fire/', 'Inyo', '25', '25', '25', '2013-02-28T20:00:00Z', '4', 'FALSE', 'TRUE', 'Unknown', '2', '36.602575', 'south of Narrow Gauge Rd & north of Hwy 136, east of Lone Pine', '-118.01651', 'TRUE', 'River Fire', '100', '476', 'The River Fire burned in February 2013, south of Narrow Gauge Rd & north of Highway 136, east of Lone Pine in Inyo County. \r\n', '2013-02-24T08:16:00Z', 'Finalized', '2013-02-28T20:00:00Z', 'Normal Fire');