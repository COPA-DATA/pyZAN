# Welcome to pyZAN

pyZAN is a python module included in the "**CopaData**" package ([PyPi](https://pypi.org/project/CopaData/)). The idea behind this module is to provide an easy-to-use interface for python users to all the data, that the zenon platform can deliver. 

zenon is a software platform for industrial digitalization , that offers:
- data acquisition using > 300 protocols (like BACNet, OPC UA, ModBus, S7 RFC 1006,...)
- calculations, rule-based events, aggregations
- alarming and event management
- data historian
- visualization
- reporting
- HMI/SCADA
- ...

## Tutorials

In this repository you can find two tutorials to get you started with pyZAN and have your first simple data science projects based on zenon data. To follow these tutorials you will need:

- zenon Software Platform 10 or higher with the components:
  - Engineering Studio
  - Service Engine
  - Report Engine
- If you are using older versions:
  - zenon Supervisor > 8.20
  - zenon Analayzer > 3.40
- basic python knowledge
- python 3 installed and these packages (install them with `pip install <package name>`):
  - `pyodbc`
  - `CopaData`
  - `seaborn` - will install these also needed dependencies:
    - `numpy`
    - `pandas`
    - `matplotlib`
  - `scikit-learn` (also contains the `sklearn` package in the samples, but `sklearn` is deprecated and has been replaced by `scikit-learn` for installation)
  - `tensorflow`
  - `keras`
- SQL Server Native Client 11.0 installed from [here](https://www.microsoft.com/en-us/download/details.aspx?id=56041) - download and install `sqlncli.msi`

Sample data is provided as CSV/NPY files, so you could follow the tutorial without the zenon environment. But to really get the gist of it, I would recommend using the provided zenon project to generate the data yourself. This can be done by following these steps:

1. Restore the project backup `zenon project -  predictive_maintenance_demo_820.zip` from the `Tutorials` folder in Engineering Studio.
2. Create a Report Engine database in Reporting Studio and use Metadata Synchronizer in Engineering Studio to fill the Report Engine database with the project metadata. The tutorials use the name `ZA_Predictive820` for the Report Engine database.
3. Start the Service Engine with the restored project and start the Service Engine Connector from the Tools tab in Startup Tool.
4. Wherever a `pyZAN.Server` is instantiated, add the constructor argument `analyzer_major_version=<your Report Engine version goes here>` to ensure the correct database structure definition version is used.
5. The samples use time filters relative to a reference time stamp for querying data from the Service Engine. Adjust the reference timestamps in order to get data from the running simulation.

Have fun!
