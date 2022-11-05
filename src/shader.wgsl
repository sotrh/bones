struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) indices: vec4<i32>,
    @location(3) weights: vec4<f32>,
}

struct VsData {
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) indices: vec4<i32>,
    @location(2) weights: vec4<f32>,
}

struct Uniforms {
    debug: vec4<f32>,
    mv: mat4x4<f32>,
    mvp: mat4x4<f32>,
}

@group(0)
@binding(0)
var<uniform> uniforms: Uniforms;

@group(0)
@binding(1)
var<storage> pose: array<mat4x4<f32>>;

@group(0)
@binding(2)
var<storage> inv_bind_pose: array<mat4x4<f32>>;

let MAX_BONES: u32 = 16u;
let MAX_INDICES: u32 = 4u;
let MODE_SHADED: u32 = 0u;
let MODE_WEIGHTS: u32 = 1u;


@vertex
fn vs_main(v: Vertex) -> VsData {
    var final_position = vec4<f32>(0.0);
    var final_normal = vec4<f32>(0.0);
    for (var i: u32 = 0u; i < MAX_INDICES; i++) {
        var bi = -1;
        if i == 0u {
            bi = v.indices[0u];
        }
        else if i == 1u {
            bi = v.indices[1u];
        }
        else if i == 2u {
            bi = v.indices[2u];
        }
        else if i == 3u {
            bi = v.indices[3u];
        }
        if bi < 0 { continue; }
        let b = u32(bi);
        let pose_mat = pose[b];
        let inv_bind_pose_mat = inv_bind_pose[b];
        let local_position: vec4<f32> = pose_mat * inv_bind_pose_mat * vec4(v.position, 1.0);
        let local_normal: vec4<f32> = pose_mat * inv_bind_pose_mat * vec4(v.normal, 0.0);

        // You can't index v.weights with i, it needs to be a constant
        var weight: f32 = 0.0;
        if i == 0u {
            weight = v.weights[0u];
        } else if i == 1u {
            weight = v.weights[1u];
        } else if i == 2u {
            weight = v.weights[2u];
        } else if i == 3u {
            weight = v.weights[3u];
        }

        final_position = final_position + local_position * weight;
        final_normal = final_normal + local_normal * weight;
    }

    // Make sure the w value is correct
    final_position.w = 1.0;
    final_normal.w = 0.0;

    // DEBUG
    // final_position = vec4(v.position, 1.0);
    // final_normal = vec4(v.normal, 0.0);

    return VsData(
        uniforms.mvp * final_position,
        normalize((uniforms.mv * final_normal).xyz),
        v.indices,
        v.weights,
    );
}

@fragment
fn fs_main(vin: VsData) -> @location(0) vec4<f32> {
    var color = vec3(0.0);
    let mode = u32(uniforms.debug.x);
    if mode == MODE_WEIGHTS {
        let selected_bone = i32(uniforms.debug.y);
        var weight = 0.0;
        if (vin.indices[0u] == selected_bone) {
            weight = vin.weights[0u];
        }
        else if (vin.indices[1u] == selected_bone) {
            weight = vin.weights[1u];
        }
        else if (vin.indices[2u] == selected_bone) {
            weight = vin.weights[2u];
        }
        else if (vin.indices[3u] == selected_bone) {
            weight = vin.weights[3u];
        }
        color = mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), weight);
    } else {
        let normal = normalize(vin.normal);
        color = vec3(clamp(dot(normal, vec3<f32>(0.0, 0.0, -1.0)), 0.0, 1.0));
    }
    return vec4(color, 1.0);
}