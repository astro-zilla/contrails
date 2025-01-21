import gzip
import shutil
import ansys.fluent.core as pyfluent
from pathlib import Path

def mesh_enclosure_pyfluent(pmdb_filename):
    meshing = pyfluent.launch_fluent(
        precision='double',
        dimension=3,
        processor_count=4,
        cleanup_on_exit=True,
        ui_mode='no_gui_or_graphics',
        mode='meshing',
        additional_arguments="-post")
    workflow = meshing.workflow

    workflow.InitializeWorkflow(WorkflowType=r'Watertight Geometry')
    workflow.TaskObject['Import Geometry'].Arguments.set_state({r'FileName': f'{pmdb_filename.absolute()}',r'ImportCadPreferences': {r'MaxFacetLength': 0,},r'LengthUnit': r'mm',})
    workflow.TaskObject['Import Geometry'].Execute()
    workflow.TaskObject['Add Local Sizing'].AddChildAndUpdate(DeferUpdate=False)
    workflow.TaskObject['Add Local Sizing'].InsertNextTask(CommandName=r'CreateLocalRefinementRegions')
    workflow.TaskObject['Create Local Refinement Regions'].Arguments.set_state({r'BOIMaxSize': 5,r'BOISizeName': r'boi_1',r'BoundingBoxObject': {r'SizeRelativeLength': r'Ratio relative to geometry size',r'Xmax': 680.0057363510132,r'XmaxRatio': 2.5,r'Xmin': -60.00275344848633,r'XminRatio': 0.2,r'Ymax': 115.470183634758,r'YmaxRatio': 0.1,r'Ymin': -6.967033410072331,r'YminRatio': 0.1,r'Zmax': 68.06188735961913,r'ZmaxRatio': 0.1,r'Zmin': -6.187444305419922,r'ZminRatio': 0.1,},r'CreationMethod': r'Box',r'CylinderObject': {r'HeightBackInc': 0,r'HeightFrontInc': 0,r'HeightNode': r'none',r'Node1': r'none',r'Node2': r'none',r'Node3': r'none',r'Options': r'3 Arc Nodes',r'Radius1': 0,r'Radius2': 0,},r'LabelSelectionList': [r'nacelle_wall', r'nose_wall', r'tail_wall'],r'OffsetObject': {r'AspectRatio': 5,r'BoundaryLayerHeight': 64,r'BoundaryLayerLevels': 1,r'CrossWakeGrowthFactor': 1.1,r'DefeaturingSize': 4,r'FirstHeight': 0.01,r'FlipDirection': False,r'FlowDirection': r'X',r'LastRatioPercentage': 20,r'MptMethodType': r'Automatic',r'NumberOfLayers': 4,r'OffsetMethodType': r'uniform',r'Rate': 1.2,r'ShowCoordinates': True,r'WakeGrowthFactor': 2,r'WakeLevels': 1,r'X': 0,r'Y': 0,r'Z': 0,},r'RefinementRegionsName': r'local-refinement-1',r'SelectionType': r'label',r'VolumeFill': r'hexcore',})
    workflow.TaskObject['Create Local Refinement Regions'].AddChildAndUpdate(DeferUpdate=False)
    workflow.TaskObject['Create Local Refinement Regions'].InsertNextTask(CommandName=r'SetUpPeriodicBoundaries')
    workflow.TaskObject['local-refinement-1'].Arguments.set_state({r'BOIMaxSize': 5,r'BOISizeName': r'local-refinement-1',r'BoundingBoxObject': {r'SizeRelativeLength': r'Ratio relative to geometry size',r'Xmax': 680.0057363510132,r'XmaxRatio': 2.5,r'Xmin': -60.00275344848633,r'XminRatio': 0.2,r'Ymax': 115.470183634758,r'YmaxRatio': 0.1,r'Ymin': -6.967033410072331,r'YminRatio': 0.1,r'Zmax': 68.06188735961913,r'ZmaxRatio': 0.1,r'Zmin': -6.187444305419922,r'ZminRatio': 0.1,},r'CreationMethod': r'Box',r'CylinderObject': {r'HeightBackInc': 0,r'HeightFrontInc': 0,r'HeightNode': r'none',r'Node1': r'none',r'Node2': r'none',r'Node3': r'none',r'Options': r'3 Arc Nodes',r'Radius1': 0,r'Radius2': 0,},r'LabelSelectionList': [r'nacelle_wall', r'nose_wall', r'tail_wall'],r'ObjectSelectionList': None,r'ObjectSelectionSingle': None,r'OffsetObject': {r'AspectRatio': 5,r'BoundaryLayerHeight': 64,r'BoundaryLayerLevels': 1,r'CrossWakeGrowthFactor': 1.1,r'DefeaturingSize': 4,r'EdgeSelectionList': None,r'FirstHeight': 0.01,r'FlipDirection': False,r'FlowDirection': r'X',r'LastRatioPercentage': 20,r'MptMethodType': r'Automatic',r'NumberOfLayers': 4,r'OffsetMethodType': r'uniform',r'Rate': 1.2,r'ShowCoordinates': True,r'WakeGrowthFactor': 2,r'WakeLevels': 1,r'X': 0,r'Y': 0,r'Z': 0,},r'RefinementRegionsName': r'local-refinement-1',r'SelectionType': r'label',r'TopologyList': None,r'VolumeFill': r'hexcore',r'ZoneLocation': None,r'ZoneSelectionList': None,r'ZoneSelectionSingle': None,})
    workflow.TaskObject['Set Up Periodic Boundaries'].Arguments.set_state({r'LCSVector': {r'VectorX': 1,},r'LabelList': [r'periodic_1'],r'PeriodicityAngle': 36,})
    workflow.TaskObject['Set Up Periodic Boundaries'].Execute()
    workflow.TaskObject['Generate the Surface Mesh'].Execute()
    workflow.TaskObject['Describe Geometry'].UpdateChildTasks(SetupTypeChanged=False)
    workflow.TaskObject['Describe Geometry'].Arguments.set_state({r'NonConformal': r'No',r'SetupType': r'The geometry consists of only fluid regions with no voids',})
    workflow.TaskObject['Describe Geometry'].UpdateChildTasks(SetupTypeChanged=True)
    workflow.TaskObject['Describe Geometry'].Execute()
    workflow.TaskObject['Update Boundaries'].Arguments.set_state({r'BoundaryLabelList': [r'zero_radius', r'stator_interface', r'rotor_interface'],r'BoundaryLabelTypeList': [r'interface', r'interface', r'interface'],r'OldBoundaryLabelList': [r'zero_radius', r'stator_interface', r'rotor_interface'],r'OldBoundaryLabelTypeList': [r'wall', r'wall', r'wall'],})
    workflow.TaskObject['Update Boundaries'].Execute()
    workflow.TaskObject['Update Regions'].Execute()
    workflow.TaskObject['Add Boundary Layers'].Arguments.set_state({r'LocalPrismPreferences': {r'Continuous': r'Continuous',},})
    workflow.TaskObject['Add Boundary Layers'].AddChildAndUpdate(DeferUpdate=False)
    workflow.TaskObject['Generate the Volume Mesh'].Arguments.set_state({r'PrismPreferences': {r'PrismAdjacentAngle': 89,r'PrismMaxAspectRatio': 100,r'PrismStairStepOptions': r'No, Exclude both checks',},r'Solver': r'CFX',r'VolumeFill': r'hexcore',r'VolumeMeshPreferences': {r'Avoid1_8Transition': r'yes',r'QualityWarningLimit': 0,},})
    workflow.TaskObject['Generate the Volume Mesh'].Execute()

    #todo do this properly (not in tui)
    # solver = meshing.tui.switch_to_solution_mode()
    solver = meshing.switch_to_solver()
    solver.file.write(file_type="case",file_name=pmdb_filename.stem+'.cas.gz')
    # solver.tui.file.write_case(pmdb_filename.stem+".cas.gz")
    solver.exit()

if __name__ == "__main__":
    geom_dir = Path("geom")
    pmdb_filename = geom_dir / "enclosure_periodic.pmdb"

    # mesh_enclosure_raw(pmdb_filename)
    mesh_enclosure_pyfluent(pmdb_filename)

    cas_gz_filename = pmdb_filename.with_suffix(".cas.gz")
    cas_filename = pmdb_filename.parent.parent/"grid"/(pmdb_filename.stem+".cas")

    print(f"converting {cas_gz_filename} to {cas_filename}...")
    # unzip cas file
    with gzip.open(cas_gz_filename, 'rb') as f_in:
        with open(cas_filename, 'wb') as f_out:
            shutil.copyfileobj(f_in,f_out)

    pmdb_filename.with_suffix(".cas.gz").unlink()


    print("done!")
