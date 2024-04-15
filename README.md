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

-  zenon Supervisor > 8.20
- zenon Analayzer > 3.40
- basic python knowledge

Sample data is provided as CSV/NPY files, so you could follow the tutorial without the zenon environment. But to really get the gist of it, I would recommend using the provided zenon project to generate the data yourself. This can be done by following these steps:

1. Restore the project backup `zenon project -  predictive_maintenance_demo_820.zip` from the `Tutorials` folder in Engineering Studio.
2. Create a Report Engine database in Reporting Studio and use Metadata Synchronizer in Engineering Studio to fill the Report Engine database with the project metadata. The tutorals use the name `ZA_Predictive820` for the Report Engine database.

Have fun!
