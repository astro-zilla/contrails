import json
import os
import tempfile
import time
import logging

import ansys.meshing.prime as prime
import ansys.meshing.prime.graphics as graphics

from setup import BoundaryCondition

t0 = time.time()


def surface_mesh(model: prime.Model):
    # start prime meshing
    io = prime.FileIO(model)

    import_params = prime.ImportCadParams(
        model, cad_reader_route=prime.CadReaderRoute.WORKBENCH,
        refacet=True, cad_refaceting_params=prime.CadRefacetingParams(
            model,prime.CadFaceter.PARASOLID,custom_normal_angle_tolerance=4)
    )



    # import cad
    print("importing CAD")
    io.import_cad("geom/nacelle.scdoc", params=import_params)
    nacelle = model.parts[0]
    print(nacelle)
    print(nacelle.get_face_zonelets())
    print(f"imported {nacelle.name} with {len(list(nacelle.get_topo_faces()))} faces at {time.time() - t0:.2f} seconds")


    # display = graphics.Graphics(model)
    # display(model.parts, update=True, scope=prime.ScopeDefinition(model, entity_type=prime.ScopeEntity.FACEZONELETS))




    # global size control
    model.set_global_sizing_params(prime.GlobalSizingParams(model, min=3.0, max=80000, growth_rate=1.2))

    # wake size control
    wake_size_control = model.control_data.create_size_control(prime.SizingType.HARD)
    wake_size_control.set_hard_sizing_params(prime.HardSizingParams(model, min=18.7))
    wake_size_control.set_scope(prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.LABELS, label_expression="wake_*er_internal"))
    print(f"created wake size control {wake_size_control.id}")

    # curvature size controls
    walls_size_control = model.control_data.create_size_control(prime.SizingType.CURVATURE)
    walls_size_control.set_curvature_sizing_params(prime.CurvatureSizingParams(model, min=3.0, max=220.0, normal_angle=4.0))
    walls_size_control.set_scope(prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.LABELS, label_expression="*_wall"))
    print(f"created wall size control {walls_size_control.id}")
    freestream_size_control = model.control_data.create_size_control(prime.SizingType.CURVATURE)
    freestream_size_control.set_curvature_sizing_params(
        prime.CurvatureSizingParams(model, min=25120, max=80000, normal_angle=18.0))
    freestream_size_control.set_scope(prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.LABELS, label_expression="freestream"))
    print(f"created freestream size control {freestream_size_control.id}")

    # proximity size control
    proximity_size_control = model.control_data.create_size_control(prime.SizingType.PROXIMITY)
    proximity_size_control.set_proximity_sizing_params(
        prime.ProximitySizingParams(model, min=3.0, max=80000, growth_rate=1.2,
                                    elements_per_gap=4, ignore_orientation=True,
                                    ignore_self_proximity=False))
    proximity_size_control.set_scope(prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.LABELS, label_expression="*_wall"))

    # compute size field
    size_field = prime.SizeField(model)
    size_field.compute_volumetric(
        [walls_size_control.id, freestream_size_control.id, wake_size_control.id, proximity_size_control.id],
        prime.VolumetricSizeFieldComputeParams(model))
    print(f"computed size field {size_field}")

    surfer = prime.Surfer(model)

    # mesh ifaces
    iface_params = prime.SurferParams(model=model, constant_size=25.0)
    iface_faces = nacelle.get_topo_faces_of_label_name_pattern("*_iface", prime.NamePatternParams(model))
    surfer.mesh_topo_faces(nacelle.id, iface_faces, iface_params)
    print(f"meshed {len(iface_faces)} iface faces")

    # mesh walls
    wall_params = prime.SurferParams(model=model, generate_quads=False, size_field_type=prime.SizeFieldType.VOLUMETRIC)
    wall_faces = nacelle.get_topo_faces_of_label_name_pattern("*_wall", prime.NamePatternParams(model))

    surfer.mesh_topo_faces(nacelle.id, wall_faces, wall_params)
    print(f"meshed {len(wall_faces)} wall faces")

    # mesh wakes
    wake_params = prime.SurferParams(model=model, generate_quads=False, size_field_type=prime.SizeFieldType.VOLUMETRIC)
    wake_faces = nacelle.get_topo_faces_of_label_name_pattern("wake_*er_internal", prime.NamePatternParams(model))
    surfer.mesh_topo_faces(nacelle.id, wake_faces, wake_params)
    print(f"meshed {len(wake_faces)} wake faces")

    # mesh freestream
    freestream_params = prime.SurferParams(model=model, size_field_type=prime.SizeFieldType.VOLUMETRIC)
    freestream_faces = nacelle.get_topo_faces_of_label_name_pattern("freestream", prime.NamePatternParams(model))
    surfer.mesh_topo_faces(nacelle.id, freestream_faces, freestream_params)
    print(f"meshed {len(freestream_faces)} freestream faces")

    # mesh other
    other_params = prime.SurferParams(model=model, size_field_type=prime.SizeFieldType.VOLUMETRIC)
    other_faces = nacelle.get_topo_faces_of_label_name_pattern("pre_wake_*,wake*_enclosure_internal",
                                                               prime.NamePatternParams(model))
    surfer.mesh_topo_faces(nacelle.id, other_faces, other_params)
    print(f"meshed {len(other_faces)} other faces")

    print(f"completed surface meshing at {time.time() - t0:.2f} seconds")

    wake_outer = "wake_outer_internal"
    wake_inner = "wake_inner_internal"
    dead = "nacelle_wall"
    freestream = "freestream"

    # compute new volumes
    print(nacelle.compute_topo_volumes(
        prime.ComputeVolumesParams(model,
                                   prime.VolumeNamingType.BYFACELABEL,
                                   prime.CreateVolumeZonesType.PERNAMESOURCE,
                                   [wake_outer, wake_inner, dead, freestream])))


def setup_volume_controls(model):
    # setup volume controls
    wake_control = model.control_data.create_volume_control()
    wake_control.set_scope(prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.ZONES,
                                                 zone_expression=f"wake_outer_internal,wake_inner_internal"))
    wake_control.set_params(prime.VolumeControlParams(model, prime.CellZoneletType.FLUID, skip_hexcore=True))
    freestream_control = model.control_data.create_volume_control()
    freestream_control.set_scope(
        prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.ZONES, zone_expression=f"freestream"))
    freestream_control.set_params(prime.VolumeControlParams(model, prime.CellZoneletType.FLUID, skip_hexcore=True))
    dead_control = model.control_data.create_volume_control()
    dead_control.set_scope(
        prime.ScopeDefinition(model, evaluation_type=prime.ScopeEvaluationType.ZONES, zone_expression=f"nacelle_wall"))
    dead_control.set_params(prime.VolumeControlParams(model, prime.CellZoneletType.DEAD))

    return [wake_control.id, freestream_control.id, dead_control.id]


def setup_bl_controls(model):
    with open("geom/nacelle.json", "r") as f:
        bcs = BoundaryCondition(**json.load(f))
    # add BLs
    volume_scope = prime.ScopeDefinition(model, entity_type=prime.ScopeEntity.VOLUME, label_expression=f"*")

    nacelle_bl = model.control_data.create_prism_control()
    nacelle_bl.set_surface_scope(prime.ScopeDefinition(model, entity_type=prime.ScopeEntity.FACEZONELETS,
                                                       label_expression="nacelle_wall,bypass_outer_wall"))
    nacelle_bl.set_volume_scope(volume_scope)
    nacelle_bl.set_growth_params(prime.PrismControlGrowthParams(model,
                                                                prime.PrismControlOffsetType.UNIFORM,
                                                                first_height=bcs.bl_nacelle_bypass.y0,
                                                                growth_rate=bcs.bl_nacelle_bypass.GR,
                                                                n_layers=bcs.bl_nacelle_bypass.n)
                                 )
    bypass_bl = model.control_data.create_prism_control()
    bypass_bl.set_surface_scope(prime.ScopeDefinition(model, entity_type=prime.ScopeEntity.FACEZONELETS,
                                                      label_expression="bypass_inner_wall,core_outer_wall"))
    bypass_bl.set_volume_scope(volume_scope)
    bypass_bl.set_growth_params(prime.PrismControlGrowthParams(model,
                                                               prime.PrismControlOffsetType.UNIFORM,
                                                               first_height=bcs.bl_bypass_core.y0,
                                                               growth_rate=bcs.bl_bypass_core.GR,
                                                               n_layers=bcs.bl_bypass_core.n)
                                )
    core_bl = model.control_data.create_prism_control()
    core_bl.set_surface_scope(
        prime.ScopeDefinition(model, entity_type=prime.ScopeEntity.FACEZONELETS, label_expression="core_inner_wall"))
    core_bl.set_volume_scope(volume_scope)
    core_bl.set_growth_params(prime.PrismControlGrowthParams(model,
                                                             prime.PrismControlOffsetType.UNIFORM,
                                                             first_height=bcs.bl_core_tail.y0,
                                                             growth_rate=bcs.bl_core_tail.GR,
                                                             n_layers=bcs.bl_core_tail.n)
                              )
    wake_bl = model.control_data.create_prism_control()
    wake_bl.set_surface_scope(
        prime.ScopeDefinition(model, entity_type=prime.ScopeEntity.FACEZONELETS, label_expression="wake_*er_internal"))
    wake_bl.set_volume_scope(volume_scope)
    wake_bl.set_growth_params(prime.PrismControlGrowthParams(model,
                                                             prime.PrismControlOffsetType.UNIFORM,
                                                             first_height=bcs.bl_wake.y0,
                                                             growth_rate=bcs.bl_wake.GR,
                                                             n_layers=bcs.bl_wake.n)
                              )
    return [nacelle_bl.id, bypass_bl.id, core_bl.id, wake_bl.id]


def volume_mesh(model, prism_control_ids, volume_control_ids):
    automesh_params = prime.AutoMeshParams(
        model,
        size_field_type=prime.SizeFieldType.VOLUMETRIC,
        volume_fill_type=prime.VolumeFillType.POLY,
        prism_control_ids=prism_control_ids,
        volume_control_ids=volume_control_ids
    )

    print(f"began volume meshing at {time.time() - t0:.2f} seconds")
    prime.AutoMesh(model).mesh(part_id=model.parts[0].id, automesh_params=automesh_params)
    print(f"completed volume meshing at {time.time() - t0:.2f} seconds")

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

with prime.launch_prime(n_procs=8,timeout=60) as prime_client:
    # prime_client = prime.launch_prime(n_procs=4, timeout=20)
    model = prime_client.model

    model.python_logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter for message output
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Add formatter to ch stream handler
    ch.setFormatter(formatter)
    model.python_logger.addHandler(ch)


    surface_mesh(model)
    prime.FileIO(model).write_pmdat("geom/nacelle.pmdat", prime.FileWriteParams(model))
    # prime.FileIO(model).read_pmdat("geom/nacelle.pmdat", prime.FileReadParams(model))

    volume_control_ids = setup_volume_controls(model)
    prism_control_ids = setup_bl_controls(model)

    
    display = graphics.Graphics(model)
    display(model.parts, update=True, scope=prime.ScopeDefinition(model,entity_type=prime.ScopeEntity.FACEZONELETS))
    print("saved nacelle.pmdat")


    volume_mesh(model, prism_control_ids, volume_control_ids)

    print(model.parts[0].get_summary(prime.PartSummaryParams(model)))
    search = prime.VolumeSearch(model)
    print(search.get_volume_quality_summary(prime.VolumeQualitySummaryParams(model)))

    prime.FileIO(model).export_fluent_case("geom/nacelle.cas", prime.ExportFluentCaseParams(model, cff_format=False))
    print("exported nacelle.cas")


    display(model.parts, update=True, scope=prime.ScopeDefinition(model, entity_type=prime.ScopeEntity.FACEZONELETS))
