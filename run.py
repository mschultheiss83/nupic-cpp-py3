import csv
import datetime
import os
import yaml

from nupic.bindings.algorithms import SpatialPooler
from nupic.bindings.algorithms import TemporalMemory
from nupic.bindings.encoders import RDSE, RDSE_Parameters
from nupic.bindings.sdr import SDR

_NUM_RECORDS = 3000
_EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
_INPUT_FILE_PATH = os.path.join(_EXAMPLE_DIR, "gymdata.csv")
_PARAMS_PATH = os.path.join(_EXAMPLE_DIR, "model.yaml")


def runHotgym(numRecords):
  with open(_PARAMS_PATH, "r") as f:
    modelParams = yaml.safe_load(f)["modelParams"]
    enParams = modelParams["sensorParams"]["encoders"]
    spParams = modelParams["spParams"]
    tmParams = modelParams["tmParams"]

  # timeOfDayEncoder = DateEncoder(
  #   timeOfDay=enParams["timestamp_timeOfDay"]["timeOfDay"])
  # weekendEncoder = DateEncoder(
  #   weekend=enParams["timestamp_weekend"]["weekend"])
  # scalarEncoder = RandomDistributedScalarEncoder(
  #   enParams["consumption"]["resolution"])

  rdseParams = RDSE_Parameters()
  rdseParams.size     = 100
  rdseParams.sparsity = .10
  rdseParams.radius   = 10
  scalarEncoder = RDSE( rdseParams )

  # encodingWidth = (timeOfDayEncoder.getWidth()
  #                  + weekendEncoder.getWidth()
  #                  + scalarEncoder.getWidth())

  encodingWidth = scalarEncoder.size

  sp = SpatialPooler(
    inputDimensions=(encodingWidth,),
    columnDimensions=(spParams["columnCount"],),
    potentialPct=spParams["potentialPct"],
    potentialRadius=encodingWidth,
    globalInhibition=spParams["globalInhibition"],
    localAreaDensity=spParams["localAreaDensity"],
    numActiveColumnsPerInhArea=spParams["numActiveColumnsPerInhArea"],
    synPermInactiveDec=spParams["synPermInactiveDec"],
    synPermActiveInc=spParams["synPermActiveInc"],
    synPermConnected=spParams["synPermConnected"],
    boostStrength=spParams["boostStrength"],
    seed=spParams["seed"],
    wrapAround=True
  )

  tm = TemporalMemory(
    columnDimensions=(tmParams["columnCount"],),
    cellsPerColumn=tmParams["cellsPerColumn"],
    activationThreshold=tmParams["activationThreshold"],
    initialPermanence=tmParams["initialPerm"],
    connectedPermanence=spParams["synPermConnected"],
    minThreshold=tmParams["minThreshold"],
    maxNewSynapseCount=tmParams["newSynapseCount"],
    permanenceIncrement=tmParams["permanenceInc"],
    permanenceDecrement=tmParams["permanenceDec"],
    predictedSegmentDecrement=0.0,
    maxSegmentsPerCell=tmParams["maxSegmentsPerCell"],
    maxSynapsesPerSegment=tmParams["maxSynapsesPerSegment"],
    seed=tmParams["seed"]
  )

  results = []
  with open(_INPUT_FILE_PATH, "r") as fin:
    reader = csv.reader(fin)
    headers = next(reader)
    next(reader)
    next(reader)

    for count, record in enumerate(reader):

      if count >= numRecords: break

      # Convert data string into Python date object.
      dateString = datetime.datetime.strptime(record[0], "%m/%d/%y %H:%M")
      # Convert data value string into float.
      consumption = float(record[1])

      # To encode, we need to provide zero-filled numpy arrays for the encoders
      # to populate.
      # timeOfDayBits = numpy.zeros(timeOfDayEncoder.getWidth())
      # weekendBits = numpy.zeros(weekendEncoder.getWidth())
      # consumptionBits = numpy.zeros(scalarEncoder.size)
      consumptionBits = SDR(scalarEncoder.size)

      # Now we call the encoders to create bit representations for each value.
      # timeOfDayEncoder.encodeIntoArray(dateString, timeOfDayBits)
      # weekendEncoder.encodeIntoArray(dateString, weekendBits)
      scalarEncoder.encode(consumption, consumptionBits)

      # Concatenate all these encodings into one large encoding for Spatial
      # Pooling.
      # encoding = numpy.concatenate(
      #   [timeOfDayBits, weekendBits, consumptionBits]
      # )
      encoding = consumptionBits

      # Create an array to represent active columns, all initially zero. This
      # will be populated by the compute method below. It must have the same
      # dimensions as the Spatial Pooler.
      # activeColumns = numpy.zeros(spParams["columnCount"])
      activeColumns = SDR(spParams["columnCount"])

      # encodingIn = numpy.uint32(encoding.dense)
      # minicolumnsOut = numpy.uint32(activeColumns.dense)
      # Execute Spatial Pooling algorithm over input space.
      # sp.compute(encodingIn, True, minicolumnsOut)
      sp.compute(encoding, True, activeColumns)
      # activeColumnIndices = numpy.nonzero(minicolumnsOut)[0]

      # Execute Temporal Memory algorithm over active mini-columns.
      # tm.compute(activeColumnIndices, learn=True)
      tm.compute(activeColumns, learn=True)

      activeCells = tm.getActiveCells()
      print(len(activeCells))
      results.append(activeCells)

    return results


if __name__ == "__main__":
  runHotgym(_NUM_RECORDS)
