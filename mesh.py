import argparse
import json
import logging
import sys
import time
from pathlib import Path

import ansys.meshing.prime as prime

from setup import BoundaryCondition

t0 = time.time()


def surface_mesh(model: prime.Model, fname: Path, wakes: bool = True):
    # start prime meshing
    io = prime.FileIO(model)

    import_params = prime.ImportCadParams(
        model, cad_reader_route=prime.CadReaderRoute.WORKBENCH, length_unit=prime.LengthUnit.MM,
        refacet=True, cad_refaceting_params=prime.CadRefacetingParams(
            model, prime.CadFaceter.PARASOLID, faceting_resolution=prime.CadRefacetingResolution.CUSTOM,
            custom_normal_angle_tolerance=1)
    )

    # import cad
    print("importing CAD")
    io.import_cad(str(fname), params=import_params)
    nacelle = model.parts[0]
    print(nacelle)
    print(nacelle.get_face_zonelets())
    print(f"imported {nacelle.name} with {len(list(nacelle.get_topo_faces()))} faces at {time.time() - t0:.2f} seconds")
    if not args.no_display:
        import ansys.meshing.prime.graphics as graphics
        display = graphics.Graphics(model)

    # global size control
    model.set_global_sizing_params(prime.GlobalSizingParams(model, min=3.0, max=80000, growth_rate=1.2))

    # wake size control
    if wakes:
        wake_size_control = model.control_data.create_size_control(prime.SizingType.HARD)
        wake_size_control.set_hard_sizing_params(prime.HardSizingParams(model, min=18.7))
        wake_size_control.set_scope(prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.LABELS,
                                                          label_expression="wake_*er_internal"))
        print(f"created wake size control {wake_size_control.id}")

    # curvature size controls
    walls_size_control = model.control_data.create_size_control(prime.SizingType.CURVATURE)
    walls_size_control.set_curvature_sizing_params(prime.CurvatureSizingParams(model, min=3.0, max=220.0, normal_angle=4.0))
    walls_size_control.set_scope(
        prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.LABELS, label_expression="*_wall"))
    print(f"created wall size control {walls_size_control.id}")
    freestream_size_control = model.control_data.create_size_control(prime.SizingType.CURVATURE)
    freestream_size_control.set_curvature_sizing_params(
        prime.CurvatureSizingParams(model, min=25120, max=80000, normal_angle=18.0))
    freestream_size_control.set_scope(
        prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.LABELS, label_expression="freestream"))
    print(f"created freestream size control {freestream_size_control.id}")

    # proximity size control
    proximity_size_control = model.control_data.create_size_control(prime.SizingType.PROXIMITY)
    proximity_size_control.set_proximity_sizing_params(
        prime.ProximitySizingParams(model, min=3.0, max=80000, growth_rate=1.2,
                                    elements_per_gap=4, ignore_orientation=True,
                                    ignore_self_proximity=False))
    proximity_size_control.set_scope(
        prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.LABELS, label_expression="* !freestream"))

    # compute size field
    size_field = prime.SizeField(model)
    size_field.compute_volumetric(
        [control.id for control in model.control_data.size_controls],
        prime.VolumetricSizeFieldComputeParams(model, enable_multi_threading=True))
    print(f"computed size field {size_field}")

    surfer = prime.Surfer(model)

    # mesh ifaces
    iface_params = prime.SurferParams(model=model, constant_size=25.0, enable_multi_threading=True)
    iface_faces = nacelle.get_topo_faces_of_label_name_pattern("*_iface", prime.NamePatternParams(model))
    print(surfer.mesh_topo_faces(nacelle.id, iface_faces, iface_params))
    print(f"meshed {len(iface_faces)} iface faces")

    # mesh walls
    wall_params = prime.SurferParams(model=model, generate_quads=False, size_field_type=prime.SizeFieldType.VOLUMETRIC,
                                     enable_multi_threading=True)
    wall_faces = nacelle.get_topo_faces_of_label_name_pattern("* !*_iface", prime.NamePatternParams(model))
    print(surfer.mesh_topo_faces(nacelle.id, wall_faces, wall_params))
    print(f"meshed {len(wall_faces)} wall faces")

    print(f"completed surface meshing at {time.time() - t0:.2f} seconds")

    prime.lucid.Mesh(model).create_zones_from_labels("*")

    wrapper = prime.Wrapper(model)

    # close_gaps_params = prime.WrapperCloseGapsParams(model,target=prime.ScopeDefinition(model),gap_size=6.0,create_new_part=False,material_point_name="nacelle")
    # create_material_points(model)
    # wrapper.close_gaps(prime.ScopeDefinition(model, label_expression="* !freestream"),close_gaps_params)

    improve_quality_params = prime.WrapperImproveQualityParams(model, resolve_intersections=True,
                                                               resolve_invalid_node_normals=True, resolve_spikes=True,
                                                               number_of_threads=2)
    wrapper.improve_quality(nacelle.id, improve_quality_params)

    # compute new volumes
    print(nacelle.compute_topo_volumes(
        prime.ComputeVolumesParams(model,
                                   prime.VolumeNamingType.BYFACELABEL,
                                   prime.CreateVolumeZonesType.PERNAMESOURCE)))
    print(nacelle)

    # volume size controls
    # wake_volume_size_control = model.control_data.create_size_control(prime.SizingType.BOI)
    # wake_volume_size_control.set_boi_sizing_params(prime.BoiSizingParams(model, max=25.0, growth_rate=1.2))
    # wake_volume_size_control.set_scope(
    #     prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.LABELS, label_expression="bypass_*,core_*,wake_outer_internal,wake*_enclosure_internal"))
    # size_field.compute_volumetric(
    #     [control.id for control in model.control_data.size_controls],
    #     prime.VolumetricSizeFieldComputeParams(model))
    # print(f"recomputed size field {size_field}")


def setup_volume_controls(model, wakes: bool = True):
    # setup volume controls

    freestream_control = model.control_data.create_volume_control()
    freestream_control.set_scope(
        prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.ZONES, zone_expression=f"freestream*"))
    freestream_control.set_params(prime.VolumeControlParams(model, prime.CellZoneletType.FLUID, skip_hexcore=True))
    dead_control = model.control_data.create_volume_control()
    dead_control.set_scope(
        prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.ZONES, zone_expression=f"nacelle_wall*"))
    dead_control.set_params(prime.VolumeControlParams(model, prime.CellZoneletType.DEAD))
    if wakes:
        wake_control = model.control_data.create_volume_control()
        wake_control.set_scope(prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.ZONES,
                                                     zone_expression=f"wake_outer_internal*,wake_inner_internal*"))
        wake_control.set_params(prime.VolumeControlParams(model, prime.CellZoneletType.FLUID, skip_hexcore=True))
        return [freestream_control.id, dead_control.id, wake_control.id]
    else:
        return [freestream_control.id, dead_control.id]


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
    # bypass_bl = model.control_data.create_prism_control()
    # bypass_bl.set_surface_scope(prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.ZONES, zone_expression="bypass_inner_wall,core_outer_wall,bypass_pylon_wall"))
    # bypass_bl.set_volume_scope(volume_scope)
    # bypass_bl.set_growth_params(prime.PrismControlGrowthParams(model,
    #                                                            prime.PrismControlOffsetType.UNIFORM,
    #                                                            first_height=bcs.bl_bypass_core.y0*1000,
    #                                                            growth_rate=bcs.bl_bypass_core.GR,
    #                                                            n_layers=bcs.bl_bypass_core.n)
    #                             )
    # core_bl = model.control_data.create_prism_control()
    # core_bl.set_surface_scope(prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.ZONES, zone_expression="core_inner_wall,core_pylon_wall"))
    # core_bl.set_volume_scope(volume_scope)
    # core_bl.set_growth_params(prime.PrismControlGrowthParams(model,
    #                                                          prime.PrismControlOffsetType.UNIFORM,
    #                                                          first_height=bcs.bl_core_tail.y0*1000,
    #                                                          growth_rate=bcs.bl_core_tail.GR,
    #                                                          n_layers=bcs.bl_core_tail.n)
    #                           )
    if wakes:
        wake_bl = model.control_data.create_prism_control()
        wake_bl.set_surface_scope(prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.ZONES,
                                                        zone_expression="wake_*er_internal"))
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
        prism=prime.PrismParams(model, stair_step=prime.PrismStairStep(model, check_proximity=False))

    )

    print(f"began volume meshing at {time.time() - t0:.2f} seconds")
    prime.AutoMesh(model).mesh(part_id=model.parts[0].id, automesh_params=automesh_params)
    print(f"completed volume meshing at {time.time() - t0:.2f} seconds")


def create_material_points(model: prime.Model, wakes: bool = True):
    model.material_point_data.create_material_point("nacelle",
                                                    [-2000, 0, 0],
                                                    prime.CreateMaterialPointParams(model, prime.MaterialPointType.DEAD))
    model.material_point_data.create_material_point("freestream",
                                                    [0, 0, 20000],
                                                    prime.CreateMaterialPointParams(model, prime.MaterialPointType.LIVE))
    if wakes:
        model.material_point_data.create_material_point("wake2",
                                                        [1000, 0, 0],
                                                        prime.CreateMaterialPointParams(model, prime.MaterialPointType.LIVE))
        model.material_point_data.create_material_point("wake1",
                                                        [0, 0, 500],
                                                        prime.CreateMaterialPointParams(model, prime.MaterialPointType.LIVE))


def main(args):
    fname = Path(args.fname)
    with (prime.launch_prime(n_procs=args.processes, timeout=60) as prime_client):

        model = prime_client.model
        print(f"default number of threads: {model.get_num_threads()}")
        model.set_num_threads(args.threads)
        print(f"new number of threads: {model.get_num_threads()}")
        if not args.no_display:
            import ansys.meshing.prime.graphics as graphics
            display = graphics.Graphics(model)

        model.python_logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)

        # Create formatter for message output
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Add formatter to ch stream handler
        ch.setFormatter(formatter)
        model.python_logger.addHandler(ch)

        surface_mesh(model, fname, not args.no_wake)
        # check surface mesh quality before proceeding to volume meshing
        summary = model.parts[0].get_summary(prime.PartSummaryParams(model))
        print("Part summary:", summary)
        surf_search = prime.SurfaceSearch(model)
        surf_quality = surf_search.get_surface_quality_summary(prime.SurfaceQualitySummaryParams(model))
        print("Surface mesh quality summary:", surf_quality)

        prime.FileIO(model).write_pmdat(str(fname.with_suffix(".pmdat")), prime.FileWriteParams(model))
        print(f"saved {str(fname.with_suffix('.pmdat'))}")
        # prime.FileIO(model).read_pmdat("geom/nacelle.pmdat", prime.FileReadParams(model))

        volume_control_ids = setup_volume_controls(model, not args.no_wake)
        prism_control_ids = setup_bl_controls(model, not args.no_wake)
        if not args.no_display:
            display(model.parts, update=True, scope=prime.ScopeDefinition(model, entity_type=prime.ScopeEntity.FACEZONELETS))
        print(prism_control_ids)
        nw = [pc for pc in model.control_data.prism_controls if pc.id == prism_control_ids[0]][0]
        print(nw.get_surface_scope())
        print(nw.get_volume_scope())
        print(nw)

        volume_mesh(model, prism_control_ids, volume_control_ids)

        # transformation_matrix = [1e-3, 0, 0, 1,0, 1e-3, 0, 1,0, 0, 1e-3, 1,0, 0, 0, 1]
        # prime.Transform(model).transform_zonelets(model.parts[0].id,model.parts[0].get_face_zonelets(),prime.TransformParams(model, transformation_matrix))
        prime.FileIO(model).export_fluent_meshing_mesh(str(fname.with_suffix('.msh')),
                                                       prime.ExportFluentMeshingMeshParams(model))
        print(f"exported {str(fname.with_suffix('.msh'))} at {time.time() - t0:.2f} seconds")

        summary = model.parts[0].get_summary(prime.PartSummaryParams(model))
        print("Part summary:", summary)
        vtool = prime.VolumeMeshTool(model)
        vtool.check_mesh(part_id=model.parts[0].id, params=prime.CheckMeshParams(model))
        vtool.improve_by_auto_node_move(model.parts[0].id, model.parts[0].get_cell_zonelets(),
                                        model.parts[0].get_face_zonelets(), prime.AutoNodeMoveParams(model))
        print("Volume mesh check summary:", vtool)
        search = prime.VolumeSearch(model)
        print("Volume mesh quality summary:", search.get_volume_quality_summary(prime.VolumeQualitySummaryParams(model)))

        if not args.no_display:
            display(model.parts, update=True, scope=prime.ScopeDefinition(model, entity_type=prime.ScopeEntity.FACEZONELETS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate a surface mesh for a geometry using Ansys Meshing Prime.")
    parser.add_argument("fname", help="path to .scdoc file")
    parser.add_argument("--no-display", action="store_true", help="do not display the mesh in a window")
    parser.add_argument("-p", "--processes", type=int, default=8, help="number of processes to use for meshing")
    parser.add_argument("-t", "--threads", type=int, default=2, help="number of threads to use for meshing")
    parser.add_argument("--no-wake", action="store_true", help="do not mesh the wake")
    args = parser.parse_args()

    main(args)
