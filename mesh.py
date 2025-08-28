import argparse
import json
import logging
import sys
import time
from pathlib import Path

import ansys.meshing.prime as prime
import ansys.fluent.core as pyfluent
import numpy as np

from setup import BoundaryCondition

t0 = time.time()


def delete_topology(model: prime.Model):
    for part in model.parts:
        if len(part.get_topo_faces()) > 0:
            part.delete_topo_entities(
                prime.DeleteTopoEntitiesParams(
                    model, delete_geom_zonelets=True, delete_mesh_zonelets=False
                )
            )


def surface_mesh(model: prime.Model, fname: Path, wakes: bool = True, periodic: bool = False):
    # start prime meshing
    io = prime.FileIO(model)

    import_params = prime.ImportCadParams(
        model, cad_reader_route=prime.CadReaderRoute.WORKBENCH, length_unit=prime.LengthUnit.MM,
        refacet=True, cad_refaceting_params=prime.CadRefacetingParams(
            model, prime.CadFaceter.PARASOLID, faceting_resolution=prime.CadRefacetingResolution.CUSTOM,
            custom_normal_angle_tolerance=1), validate_shared_topology=True
    )

    # import cad
    print(f"importing {fname.absolute()} at {time.time() - t0:.2f} seconds")
    io.import_cad(str(fname.absolute()), params=import_params)

    # import ansys.meshing.prime.graphics as graphics
    # display = graphics.Graphics(model)
    # display(model.parts, update=True, scope=prime.ScopeDefinition(model, entity_type=prime.ScopeEntity.FACEZONELETS))
    # exit(0)
    nacelle = model.parts[0]
    print(nacelle.get_topo_faces_of_label_name_pattern("freestream", prime.NamePatternParams(model)))
    print(nacelle)
    print(nacelle.get_face_zonelets())
    print(f"imported {fname.name} with {len(list(nacelle.get_topo_faces()))} faces at {time.time() - t0:.2f} seconds")

    # global size control
    model.set_global_sizing_params(prime.GlobalSizingParams(model, min=3.0, max=80000, growth_rate=1.2))

    # wake size control
    if wakes:
        # wake_size_control = model.control_data.create_size_control(prime.SizingType.HARD)
        # wake_size_control.set_hard_sizing_params(prime.HardSizingParams(model, min=18.7))
        # wake_size_control.set_scope(prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.LABELS,
        #                                                   label_expression="wake_*er_internal"))
        # print(f"created wake size control {wake_size_control.id}")

        wake_boi_control = model.control_data.create_size_control(prime.SizingType.BOI)
        wake_boi_control.set_boi_sizing_params(prime.BoiSizingParams(model,max=18.7,growth_rate=1.2))
        wake_boi_control.set_scope(prime.ScopeDefinition(model, entity_type=prime.ScopeEntity.FACEANDEDGEZONELETS,
                                                         evaluation_type=prime.ScopeEvaluationType.LABELS,
                                                        label_expression="*internal"))
        print(f"created wake BOI control {wake_boi_control.id}")

    if periodic:
        periodic_control = model.control_data.create_periodic_control()
        periodic_control.set_params(prime.PeriodicControlParams(model, center=[0, 0, 0], axis=[1, 0, 0], angle=30.0))
        periodic_control.set_scope(prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.LABELS,
                                                         label_expression="*periodic*"))
        print(f"created periodic control {periodic_control.id}")
        print(periodic_control.get_summary(prime.PeriodicControlSummaryParams(model)).message)

    iface_size_control = model.control_data.create_size_control(prime.SizingType.HARD)
    iface_size_control.set_hard_sizing_params(prime.HardSizingParams(model,min=25.0))
    iface_size_control.set_scope(prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.LABELS,
                                                       label_expression="*iface"))
    print(f"created iface size control {iface_size_control.id}")

    # curvature size controls
    walls_size_control = model.control_data.create_size_control(prime.SizingType.CURVATURE)
    walls_size_control.set_curvature_sizing_params(
        prime.CurvatureSizingParams(model, min=3.0, max=220.0, normal_angle=4.0, use_cad_curvature=True))
    walls_size_control.set_scope(
        prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.LABELS,
                              label_expression=f"*_wall,zero_rad"))
    print(f"created wall size control {walls_size_control.id}")
    freestream_size_control = model.control_data.create_size_control(prime.SizingType.CURVATURE)
    freestream_size_control.set_curvature_sizing_params(
        prime.CurvatureSizingParams(model, min=25120, max=80000, normal_angle=18.0, use_cad_curvature=True))
    freestream_size_control.set_scope(
        prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.LABELS, label_expression="freestream"))
    print(f"created freestream size control {freestream_size_control.id}")

    # proximity size control
    proximity_size_control = model.control_data.create_size_control(prime.SizingType.PROXIMITY)
    proximity_size_control.set_proximity_sizing_params(
        prime.ProximitySizingParams(model, min=3.0, max=80000, growth_rate=1.2,
                                    elements_per_gap=4, ignore_orientation=True,
                                    ignore_self_proximity=False))
    label_expr = f"*_wall,*_iface{',wake_*er_internal' if wakes else ''}"
    proximity_size_control.set_scope(
        prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.LABELS, label_expression=label_expr)
    )
    print(f"created proximity size control {proximity_size_control.id}")

    # compute size field
    size_field = prime.SizeField(model)
    size_field.compute_volumetric(
        [control.id for control in model.control_data.size_controls],
        prime.VolumetricSizeFieldComputeParams(model, enable_multi_threading=True))
    print(f"computed size field {size_field}")
    prime.FileIO(model).write_size_field(str(fname.with_suffix('.psf')), prime.WriteSizeFieldParams(model, False))
    print(f"exported size field {str(fname.with_suffix('.psf'))} at {time.time() - t0:.2f} seconds")




    surfer = prime.Surfer(model)

    # mesh ifaces
    # iface_params = prime.SurferParams(model=model, constant_size=25.0, enable_multi_threading=True)
    # iface_faces = nacelle.get_topo_faces_of_label_name_pattern("*_iface", prime.NamePatternParams(model))
    # print(surfer.mesh_topo_faces(nacelle.id, iface_faces, iface_params))
    # print(f"meshed {len(iface_faces)} iface faces: {iface_faces}")

    # mesh walls
    wall_params = prime.SurferParams(model=model, generate_quads=False, size_field_type=prime.SizeFieldType.VOLUMETRIC,
                                     enable_multi_threading=True)
    wall_faces = nacelle.get_topo_faces_of_label_name_pattern(f"*_wall,*_iface,freestream{',zero_rad' if periodic else ''}",
                                                              prime.NamePatternParams(model))
    print(surfer.mesh_topo_faces(nacelle.id, wall_faces, wall_params))
    print(f"meshed {len(wall_faces)} wall faces: {wall_faces}")

    if wakes and 1==2:
        # mesh wake faces
        wake_params = prime.SurferParams(model=model, generate_quads=True,
                                         size_field_type=prime.SizeFieldType.VOLUMETRIC,
                                         enable_multi_threading=True)
        wake_faces = nacelle.get_topo_faces_of_label_name_pattern("*internal", prime.NamePatternParams(model))
        print(surfer.mesh_topo_faces(nacelle.id, wake_faces, wake_params))
        print(f"meshed {len(wake_faces)} wake faces: {wake_faces}")

    # mesh periodics
    if periodic:
        periodic_faces = nacelle.get_topo_faces_of_label_name_pattern("*periodic*", prime.NamePatternParams(model))
        periodic_params = prime.SurferParams(model=model, generate_quads=False,
                                             size_field_type=prime.SizeFieldType.VOLUMETRIC,
                                             enable_multi_threading=True)
        print(surfer.mesh_topo_faces(nacelle.id, periodic_faces, periodic_params))
        print(f"meshed {len(periodic_faces)} periodic faces: {periodic_faces}")

    print(f"completed surface meshing at {time.time() - t0:.2f} seconds")

    prime.lucid.Mesh(model).create_zones_from_labels("*")
    nacelle.delete_topo_entities(prime.DeleteTopoEntitiesParams(model, delete_geom_zonelets=True))

    # compute new volumes

    print(nacelle.compute_closed_volumes(
        prime.ComputeVolumesParams(model,
                                   prime.VolumeNamingType.BYFACELABEL,
                                   prime.CreateVolumeZonesType.PERNAMESOURCE, priority_ordered_names=["freestream"])))

    print(nacelle)


def setup_volume_controls(model, wakes: bool, periodic: bool):
    # setup volume controls

    freestream_control = model.control_data.create_volume_control()
    freestream_control.set_scope(
        prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.ZONES, zone_expression=f"freestream*"))
    freestream_control.set_params(prime.VolumeControlParams(model, prime.CellZoneletType.FLUID, skip_hexcore=True))
    if not periodic:
        dead_control = model.control_data.create_volume_control()
        dead_control.set_scope(
            prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.ZONES, zone_expression=f"nacelle_wall*"))
        dead_control.set_params(prime.VolumeControlParams(model, prime.CellZoneletType.DEAD))
        if wakes and 1==2:
            wake_control = model.control_data.create_volume_control()
            wake_control.set_scope(prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.ZONES,
                                                         zone_expression=f"wake_outer_internal*,wake_inner_internal*"))
            wake_control.set_params(prime.VolumeControlParams(model, prime.CellZoneletType.FLUID, skip_hexcore=False))
            return [freestream_control.id, dead_control.id, wake_control.id]
        else:
            return [freestream_control.id, dead_control.id]
    else:
        return [freestream_control.id]


def setup_bl_controls(model, wakes: bool = True):
    with open("geom/nacelle.json", "r") as f:
        bcs = BoundaryCondition(**json.load(f))
    # add BLs
    volume_scope = prime.ScopeDefinition(model, entity_type=prime.ScopeEntity.VOLUME)

    nacelle_bl = model.control_data.create_prism_control()
    nacelle_bl.set_surface_scope(
        prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.ZONES, zone_expression="*_wall"))
    nacelle_bl.set_volume_scope(volume_scope)
    nacelle_bl.set_growth_params(prime.PrismControlGrowthParams(model,
                                                                prime.PrismControlOffsetType.UNIFORM,
                                                                first_height=bcs.bl_nacelle_bypass.y0 * 1000,
                                                                growth_rate=bcs.bl_nacelle_bypass.GR,
                                                                n_layers=bcs.bl_nacelle_bypass.n)
                                 )

    if wakes and 1 == 2:
        wake_bl = model.control_data.create_prism_control()
        wake_bl.set_surface_scope(prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.ZONES,
                                                        zone_expression="*internal*"))
        wake_bl.set_volume_scope(volume_scope)
        wake_bl.set_growth_params(prime.PrismControlGrowthParams(model,
                                                                 prime.PrismControlOffsetType.UNIFORM,
                                                                 first_height=bcs.bl_wake.y0 * 1000,
                                                                 growth_rate=bcs.bl_wake.GR,
                                                                 n_layers=bcs.bl_wake.n)
                                  )
        return [nacelle_bl.id, wake_bl.id]
    else:
        return [nacelle_bl.id]


def volume_mesh(model, prism_control_ids, volume_control_ids):
    automesh_params = prime.AutoMeshParams(
        model,
        size_field_type=prime.SizeFieldType.VOLUMETRIC,
        volume_fill_type=prime.VolumeFillType.TET,
        prism_control_ids=prism_control_ids,
        volume_control_ids=volume_control_ids,
        periodic_control_ids=[p.id for p in model.control_data.periodic_controls],
        multi_zone_control_ids=[p.id for p in model.control_data.multi_zone_controls()],
        prism=prime.PrismParams(model, stair_step=prime.PrismStairStep(model, check_proximity=False)),
        hexcore=prime.HexCoreParams(model,
                                    transition_size_field_type=prime.SizeFieldType.VOLUMETRIC,
                                    buffer_layers=2,
                                    transition_layer_type=prime.HexCoreTransitionLayerType.DELETELARGE,
                                    cell_element_type=prime.HexCoreCellElementType.ALLHEX,
                                    enable_region_based_hexcore=True)

    )

    print(f"began volume meshing at {time.time() - t0:.2f} seconds")
    prime.AutoMesh(model).mesh(part_id=model.parts[0].id, automesh_params=automesh_params)
    print(f"completed volume meshing at {time.time() - t0:.2f} seconds")


def create_material_points(model: prime.Model, wakes: bool = True):
    # model.material_point_data.create_material_point("nacelle",
    #                                                 [-2000, 0, 0],
    #                                                 prime.CreateMaterialPointParams(model, prime.MaterialPointType.DEAD))
    model.material_point_data.create_material_point("freestream",
                                                    [0, 10000, 20],
                                                    prime.CreateMaterialPointParams(model, prime.MaterialPointType.LIVE))
    if wakes:
        model.material_point_data.create_material_point("wake1",
                                                        [2000, 250, 0],
                                                        prime.CreateMaterialPointParams(model, prime.MaterialPointType.LIVE))
        model.material_point_data.create_material_point("wake2",
                                                        [2000, 600, 0],
                                                        prime.CreateMaterialPointParams(model, prime.MaterialPointType.LIVE))


def main(args):
    fname = Path(args.fname)
    with (prime.launch_prime(n_procs=args.processes, timeout=60) as prime_client):

        model = prime_client.model
        model.set_num_threads(args.threads)

        print(f"launched with {args.processes} processes and {model.get_num_threads()} threads")

        if not args.no_display:
            import ansys.meshing.prime.graphics as graphics
            display = graphics.Graphics(model)


        if args.verbose:
            model.python_logger.setLevel(logging.INFO)
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.INFO)
        elif args.very_verbose:
            model.python_logger.setLevel(logging.DEBUG)
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.DEBUG)
        else:
            model.python_logger.setLevel(logging.WARNING)
            ch = logging.StreamHandler(stream=sys.stderr)
            ch.setLevel(logging.WARNING)

        # Create formatter for message output
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Add formatter to ch stream handler
        ch.setFormatter(formatter)
        model.python_logger.addHandler(ch)

        # begin meshing
        create_material_points(model, False)
        surface_mesh(model, fname, not args.no_wake, "periodic" in fname.name)
        # check surface mesh quality before proceeding to volume meshing
        summary = model.parts[0].get_summary(prime.PartSummaryParams(model))
        print("Part summary:", summary)
        surf_search = prime.SurfaceSearch(model)
        surf_quality = surf_search.get_surface_quality_summary(prime.SurfaceQualitySummaryParams(model))
        print("Surface mesh quality summary:", surf_quality)

        prime.FileIO(model).write_pmdat(str(fname.with_suffix(".pmdat")), prime.FileWriteParams(model))
        print(f"saved {str(fname.with_suffix('.pmdat'))}")
        # prime.FileIO(model).read_pmdat("geom/nacelle.pmdat", prime.FileReadParams(model))

        volume_control_ids = setup_volume_controls(model, not args.no_wake, "periodic" in fname.name)
        prism_control_ids = setup_bl_controls(model, not args.no_wake)
        if not args.no_display:
            display(model.parts, update=True, scope=prime.ScopeDefinition(model, entity_type=prime.ScopeEntity.FACEZONELETS))
        print(prism_control_ids)

        volume_mesh(model, prism_control_ids, volume_control_ids)

        # transformation_matrix = [1e-3, 0, 0, 1,0, 1e-3, 0, 1,0, 0, 1e-3, 1,0, 0, 0, 1]
        # prime.Transform(model).transform_zonelets(model.parts[0].id,model.parts[0].get_face_zonelets(),prime.TransformParams(model, transformation_matrix))
        prime.FileIO(model).export_fluent_meshing_mesh(str(fname.with_suffix('.msh')),
                                                       prime.ExportFluentMeshingMeshParams(model))
        print(f"exported {str(fname.with_suffix('.msh'))} at {time.time() - t0:.2f} seconds")

        summary = model.parts[0].get_summary(prime.PartSummaryParams(model))
        print("Part summary:", summary)
        vtool = prime.VolumeMeshTool(model)
        vtool.improve_by_auto_node_move(model.parts[0].id, model.parts[0].get_cell_zonelets(),
                                        model.parts[0].get_face_zonelets(), prime.AutoNodeMoveParams(model))
        vtool.check_mesh(part_id=model.parts[0].id, params=prime.CheckMeshParams(model))
        print("Volume mesh check summary:", vtool)
        search = prime.VolumeSearch(model)
        print("Volume mesh quality summary:", search.get_volume_quality_summary(prime.VolumeQualitySummaryParams(model)))

        prime.FileIO(model).export_fluent_meshing_mesh(str(fname.with_suffix('.msh')),
                                                       prime.ExportFluentMeshingMeshParams(model))
        print(f"exported {str(fname.with_suffix('.msh'))} at {time.time() - t0:.2f} seconds")

        prime.FileIO(model).write_size_field(str(fname.with_suffix('.psf')), prime.WriteSizeFieldParams(model, False))

        if not args.no_display:
            display(model.parts, update=True, scope=prime.ScopeDefinition(model, entity_type=prime.ScopeEntity.FACEZONELETS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate a surface mesh for a geometry using Ansys Meshing Prime.")
    parser.add_argument("fname", help="path to .scdoc file")
    parser.add_argument("--no-display", action="store_true", help="do not display the mesh in a window")
    parser.add_argument("-p", "--processes", type=int, default=8, help="number of processes to use for meshing")
    parser.add_argument("-t", "--threads", type=int, default=2, help="number of threads to use for meshing")
    parser.add_argument("--no-wake", action="store_true", help="do not mesh the wake")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable verbose output")
    parser.add_argument("-vv", "--very-verbose", action="store_true", help="enable very verbose output")

    args = parser.parse_args()

    main(args)
