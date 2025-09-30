import time
from pathlib import Path

from ansys.meshing import prime

fname = Path("geom/shear.scdoc")
t0=time.time()

with (prime.launch_prime(n_procs=16, timeout=60) as prime_client):
    model = prime_client.model
    model.set_num_threads(2)

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
    part = model.parts[0]
    print(f"imported {fname.name} with {len(list(part.get_topo_faces()))} faces at {time.time() - t0:.2f} seconds")
    summary = model.parts[0].get_summary(prime.PartSummaryParams(model))
    print("Part summary:", summary)

    # global size control
    model.set_global_sizing_params(prime.GlobalSizingParams(model, min=10.0, max=500, growth_rate=1+1/500))


    # periodic_control = model.control_data.create_periodic_control()
    # periodic_control.set_params(prime.PeriodicControlParams(model))
    # periodic_control.set_scope(prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.LABELS,
    #                                                  label_expression="*_periodic"))
    # print(f"created periodic control {periodic_control.id}")

    # curvature size controls
    splitter_control = model.control_data.create_size_control(prime.SizingType.HARD)
    splitter_control.set_scope(prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.LABELS,
                                                        label_expression="splitter*"))
    splitter_control.set_hard_sizing_params(prime.HardSizingParams(model, min=10.0))
    print(f"created splitter size control {splitter_control.id} at {time.time() - t0:.2f} seconds")

    # wake_control = model.control_data.create_size_control(prime.SizingType.SOFT)
    # wake_control.set_scope(prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.LABELS,
    #                                                     label_expression="wake*"))
    # wake_control.set_soft_sizing_params(prime.SoftSizingParams(model, max=20.0, growth_rate=1.05))
    # print(f"created wake size control {wake_control.id} at {time.time() - t0:.2f} seconds")

    # proximity_control = model.control_data.create_size_control(prime.SizingType.PROXIMITY)
    # proximity_control.set_scope(prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.LABELS,
    #                                                     label_expression="splitter*"))
    # proximity_control.set_proximity_sizing_params(prime.ProximitySizingParams(model, min=10, max=500, growth_rate=1+1/500))
    # print(f"created proximity size control {proximity_control.id} at {time.time() - t0:.2f} seconds")

    # compute size field
    size_field = prime.SizeField(model)
    res=size_field.compute_volumetric(
        [control.id for control in model.control_data.size_controls],
        prime.VolumetricSizeFieldComputeParams(model, enable_multi_threading=True))
    print(f"computed size field:\n{res}")

    surfer = prime.Surfer(model)

    # mesh walls
    wall_params = prime.SurferParams(model=model, generate_quads=False, size_field_type=prime.SizeFieldType.VOLUMETRIC,
                                     enable_multi_threading=True)
    wall_faces = part.get_topo_faces_of_label_name_pattern(f"*",
                                                           prime.NamePatternParams(model))
    print(surfer.mesh_topo_faces(part.id, wall_faces, wall_params))
    print(f"meshed {len(wall_faces)} wall faces: {wall_faces}")
    print(f"completed surface meshing at {time.time() - t0:.2f} seconds")

    prime.lucid.Mesh(model).create_zones_from_labels("*")
    part.delete_topo_entities(prime.DeleteTopoEntitiesParams(model, delete_geom_zonelets=True))

    # compute new volumes

    print(part.compute_closed_volumes(
        prime.ComputeVolumesParams(model,
                                   prime.VolumeNamingType.BYFACELABEL,
                                   prime.CreateVolumeZonesType.PERNAMESOURCE, priority_ordered_names=["freestream"])))

    freestream_control = model.control_data.create_volume_control()
    freestream_control.set_scope(
        prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.ZONES, zone_expression=f"freestream*"))
    freestream_control.set_params(prime.VolumeControlParams(model, prime.CellZoneletType.FLUID, skip_hexcore=True))

    automesh_params = prime.AutoMeshParams(
            model,
            size_field_type=prime.SizeFieldType.VOLUMETRIC,
            volume_fill_type=prime.VolumeFillType.TET,
            prism_control_ids=[p.id for p in model.control_data.prism_controls],
            volume_control_ids=[p.id for p in model.control_data.volume_controls],
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

    prime.FileIO(model).export_fluent_meshing_mesh(str(fname.with_suffix('.msh')),
                                                       prime.ExportFluentMeshingMeshParams(model))
    print(f"exported {str(fname.with_suffix('.msh'))} at {time.time() - t0:.2f} seconds")