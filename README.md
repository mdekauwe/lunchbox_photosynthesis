# Lunchbox Photosynthesis

Code for sandwich box photosynthesis logger.


<p float="left">
  <img src="img/IMG_6177.jpg" width="350" />
  <img src="img/plot.JPG" width="450" />
</p>

## Notes

- Box screens about ~15% of PAR (testing with licor PAR sensor).
- The SCD40 sensor has an auto calibration on CO2, but requires it to be outside for some period of time. Ultimately this doesn't matter if logging relative change, but worth noting. I tested forcing the CO2 to a new minimum, but this doesn't really work.
- The SCD40's temperature logger is biased high when in direct light. This is probably also the "greenhouse effect" of the plastic box. I suspect if we add a "Stevenson screen", i.e. a piece of white yogurt pot, with holes, this might be sufficient. We could test this by putting two sensors in a box and comparing readings. Currently, this does have an impact because of how we're calculating A_net. I have bought some insolation tape to test that
- Adding a fan has mixed results. It does lead to higher measured values, but it looks like you need to pulse things (turn it on and off). If it is too close to the sensor and I think it ends up blowing moisture onto the sensor as the RH goes to 100%. Going to test moving the fan a long way from the sensor and to box off the sensor.
- Using the Xensiv PAS CO2 sensor leads to a lack of precision as it will only return the CO2 concentration as two bytes (MSB and LSB), coded as a signed 16bit integer with a resolution of 1 ppm per bit.
