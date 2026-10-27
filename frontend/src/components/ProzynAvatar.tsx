import {
  Suspense,
  useMemo,
  useRef,
} from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";

export type AvatarState =
  | "ready"
  | "listening"
  | "thinking"
  | "speaking";

export type Identity = "sara" | "alex";

type PointData = {
  positions: Float32Array;
  brightness: Float32Array;
};

type MorphData = {
  sara: PointData;
  alex: PointData;
};

type Props = {
  identity: Identity;
  state: AvatarState;
  getLevel?: () => number;
};

const MODELS = {
  sara: "/models/sara.glb",
  alex: "/models/alex.glb",
};

const PARTICLE_COUNT = 16000;
const TARGET_HEIGHT = 2.2;

const SARA_COLOR = new THREE.Color("#73d9ff");
const ALEX_COLOR = new THREE.Color("#70a8ff");
const HIGHLIGHT_COLOR = new THREE.Color("#d9f5ff");

function seededRandom(seed: number) {
  let value = seed | 0;

  return () => {
    value = (value * 1664525 + 1013904223) | 0;
    return ((value >>> 0) / 4294967296);
  };
}

function collectMeshVertices(root: THREE.Object3D) {
  root.updateMatrixWorld(true);

  const vertices: number[] = [];

  root.traverse((object) => {
    const mesh = object as THREE.Mesh;

    if (!mesh.isMesh || !mesh.geometry) {
      return;
    }

    const geometry = mesh.geometry.index
      ? mesh.geometry.toNonIndexed()
      : mesh.geometry.clone();

    geometry.applyMatrix4(mesh.matrixWorld);

    const position = geometry.getAttribute("position");

    for (let i = 0; i < position.count; i += 1) {
      vertices.push(
        position.getX(i),
        position.getY(i),
        position.getZ(i),
      );
    }

    geometry.dispose();
  });

  return new Float32Array(vertices);
}

function normalizePoints(source: Float32Array) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;

  for (let i = 0; i < source.length; i += 3) {
    let x = source[i];
    const y = source[i + 1];
    const z = source[i + 2];

    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);

    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);

    minZ = Math.min(minZ, z);
    maxZ = Math.max(maxZ, z);
  }

  const height = Math.max(maxY - minY, 0.0001);
  const scale = TARGET_HEIGHT / height;

  const centerX = (minX + maxX) / 2;
  const centerZ = (minZ + maxZ) / 2;

  const normalized = new Float32Array(source.length);

  for (let i = 0; i < source.length; i += 3) {
    normalized[i] = (source[i] - centerX) * scale;

    normalized[i + 1] =
      (source[i + 1] - minY - height / 2) * scale;

    normalized[i + 2] =
      (source[i + 2] - centerZ) * scale;
  }

  return normalized;
}

function samplePointCloud(
  source: Float32Array,
  count: number,
  seed: number,
  identity: Identity,
): PointData {
  const normalized = normalizePoints(source);
  const random = seededRandom(seed);

  /*
   * collectMeshVertices() currently converts indexed geometry
   * to non-indexed geometry. Therefore every 9 consecutive
   * values represent one triangle:
   *
   * [ax, ay, az, bx, by, bz, cx, cy, cz]
   *
   * We use the actual triangle surface area as the sampling
   * probability. This produces a much more uniform particle
   * distribution than randomly selecting mesh vertices.
   */
  const triangleCount =
    Math.floor(normalized.length / 9);

  if (triangleCount === 0) {
    throw new Error(
      "Avatar model contains no triangles.",
    );
  }

  const cumulativeAreas =
    new Float64Array(triangleCount);

  let totalArea = 0;

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const offset = triangle * 9;

    const ax = normalized[offset];
    const ay = normalized[offset + 1];
    const az = normalized[offset + 2];

    const bx = normalized[offset + 3];
    const by = normalized[offset + 4];
    const bz = normalized[offset + 5];

    const cx = normalized[offset + 6];
    const cy = normalized[offset + 7];
    const cz = normalized[offset + 8];

    const abX = bx - ax;
    const abY = by - ay;
    const abZ = bz - az;

    const acX = cx - ax;
    const acY = cy - ay;
    const acZ = cz - az;

    const crossX =
      abY * acZ -
      abZ * acY;

    const crossY =
      abZ * acX -
      abX * acZ;

    const crossZ =
      abX * acY -
      abY * acX;

    const area =
      0.5 *
      Math.sqrt(
        crossX * crossX +
          crossY * crossY +
          crossZ * crossZ,
      );

    /*
     * Give the facial surface a higher sampling probability.
     *
     * The original model geometry already contains the eyes,
     * nose and mouth. We only redistribute particles toward
     * the front facial region; the geometry itself is unchanged.
     */
    const centroidX =
      (ax + bx + cx) / 3;

    const centroidY =
      (ay + by + cy) / 3;

    const centroidZ =
      (az + bz + cz) / 3;

    const faceFocus =
      Math.exp(
        -(
          centroidX * centroidX * 5.0 +
          (centroidY - 0.34) *
            (centroidY - 0.34) *
            6.0
        ),
      );

    const frontFocus =
      THREE.MathUtils.smoothstep(
        centroidZ,
        0.02,
        0.34,
      );

    const faceSamplingBoost =
      1.0 +
      faceFocus *
      frontFocus *
      2.25;

    const weightedArea =
      Math.max(area, 0.00000001) *
      faceSamplingBoost;

    totalArea += weightedArea;
    cumulativeAreas[triangle] = totalArea;
  }

  const positions = new Float32Array(
    count * 3,
  );

  const brightness = new Float32Array(
    count,
  );

  for (let i = 0; i < count; i += 1) {
    /*
     * Select a triangle proportional to its surface area.
     */
    const target =
      random() * totalArea;

    let low = 0;
    let high = triangleCount - 1;

    while (low < high) {
      const middle =
        Math.floor((low + high) / 2);

      if (
        cumulativeAreas[middle] <
        target
      ) {
        low = middle + 1;
      } else {
        high = middle;
      }
    }

    const triangle = low;
    const offset = triangle * 9;

    const ax = normalized[offset];
    const ay = normalized[offset + 1];
    const az = normalized[offset + 2];

    const bx = normalized[offset + 3];
    const by = normalized[offset + 4];
    const bz = normalized[offset + 5];

    const cx = normalized[offset + 6];
    const cy = normalized[offset + 7];
    const cz = normalized[offset + 8];

    /*
     * Uniform barycentric sampling.
     *
     * sqrt(r1) prevents particles from clustering
     * around the first vertex of each triangle.
     */
    const r1 = Math.sqrt(random());
    const r2 = random();

    const weightA = 1 - r1;
    const weightB = r1 * (1 - r2);
    const weightC = r1 * r2;

    let x =
      ax * weightA +
      bx * weightB +
      cx * weightC;

    const y =
      ay * weightA +
      by * weightB +
      cy * weightC;

    let z =
      az * weightA +
      bz * weightB +
      cz * weightC;

    /*
     * SARA gets a restrained feminine silhouette:
     * slightly narrower shoulders and a softer neck width.
     * This is intentionally subtle so the original facial
     * geometry remains recognizable.
     */
    if (identity === "sara") {
      const shoulderT =
        1.0 -
        THREE.MathUtils.smoothstep(
          y,
          -0.86,
          -0.34,
        );

      const neckT =
        THREE.MathUtils.smoothstep(
          y,
          -0.42,
          0.02,
        ) *
        (1.0 -
          THREE.MathUtils.smoothstep(
            y,
            0.02,
            0.42,
          ));

      const feminineScale =
        1.0 -
        shoulderT * 0.085 -
        neckT * 0.045;

      x *= feminineScale;
    }

    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    /*
     * Keep the original restrained lighting logic.
     * Sampling changes the geometry distribution only;
     * it should not make the avatar brighter or whiter.
     */
    const headFactor =
      THREE.MathUtils.smoothstep(
        y,
        0.15,
        0.85,
      );

    const faceFactor =
      Math.exp(
        -(
          x * x * 2.2 +
          (y - 0.35) *
            (y - 0.35) *
            3.0 +
          (z - 0.2) *
            (z - 0.2) *
            3.5
        ),
      );

    brightness[i] =
      THREE.MathUtils.clamp(
        0.16 +
          headFactor * 0.16 +
          faceFactor * 0.10 +
          random() * 0.12,
        0,
        0.62,
      );
  }

  /*
   * Stable spatial ordering keeps the SARA/ALEX morph
   * deterministic and visually coherent.
   */
  const indices = Array.from(
    { length: count },
    (_, index) => index,
  );

  indices.sort((a, b) => {
    const ay = positions[a * 3 + 1];
    const by = positions[b * 3 + 1];

    if (Math.abs(ay - by) > 0.025) {
      return ay - by;
    }

    const az = positions[a * 3 + 2];
    const bz = positions[b * 3 + 2];

    if (Math.abs(az - bz) > 0.025) {
      return az - bz;
    }

    return (
      positions[a * 3] -
      positions[b * 3]
    );
  });

  const sortedPositions =
    new Float32Array(count * 3);

  const sortedBrightness =
    new Float32Array(count);

  indices.forEach(
    (sourceIndex, targetIndex) => {
      sortedPositions.set(
        positions.subarray(
          sourceIndex * 3,
          sourceIndex * 3 + 3,
        ),
        targetIndex * 3,
      );

      sortedBrightness[targetIndex] =
        brightness[sourceIndex];
    },
  );

  return {
    positions: sortedPositions,
    brightness: sortedBrightness,
  };
}

const BODY_PARTICLE_COUNT = 12000;
const HAIR_PARTICLE_COUNT =
  PARTICLE_COUNT - BODY_PARTICLE_COUNT;

function createProceduralHair(
  identity: Identity,
  count: number,
  seed: number,
): PointData {
  const random = seededRandom(seed);

  const positions = new Float32Array(
    count * 3,
  );

  const brightness = new Float32Array(
    count,
  );

  for (let i = 0; i < count; i += 1) {
    const u = random();
    const v = random();
    const w = random();

    /*
     * Hair is generated as a compact scalp shell.
     *
     * SARA gets a softer, longer silhouette with
     * side strands extending toward the jaw.
     *
     * ALEX gets a shorter, tighter and more
     * structured top hairstyle.
     */
    let x: number;
    let y: number;
    let z: number;

    if (identity === "sara") {
      /*
       * SARA: soft layered hair.
       *
       * Most particles form the scalp/crown. Additional particles
       * create long side curtains and a subtle back layer so the
       * silhouette reads as feminine even at a distance.
       */
      const angle =
        u * Math.PI * 2;

      const topBias =
        Math.pow(v, 0.70);

      const radiusX =
        0.285 +
        topBias * 0.060;

      const radiusZ =
        0.185 +
        topBias * 0.050;

      x =
        Math.cos(angle) *
        radiusX *
        (0.80 + w * 0.20);

      y =
        0.70 +
        topBias * 0.43;

      z =
        Math.sin(angle) *
        radiusZ *
        (0.80 + w * 0.20);

      /*
       * Long side curtains. Keep the front-center area open
       * so the face remains clearly visible.
       */
      if (u < 0.30) {
        const side =
          u < 0.15 ? -1 : 1;

        const strandT =
          random();

        x =
          side *
          (
            0.285 +
            strandT * 0.085
          );

        y =
          -0.18 +
          strandT * 1.02;

        z =
          -0.045 +
          random() * 0.13;

        /*
         * Slight inward taper near the lower ends.
         */
        x *=
          0.94 +
          0.06 *
          THREE.MathUtils.smoothstep(
            y,
            -0.18,
            0.82,
          );
      }

      /*
       * A small back curtain gives the hair volume without
       * covering the face.
       */
      if (u >= 0.30 && u < 0.43) {
        x =
          (random() - 0.5) *
          0.62;

        y =
          -0.12 +
          random() * 0.88;

        z =
          -0.18 -
          random() * 0.07;
      }

      /*
       * Open the front-facing facial area.
       */
      if (z > 0.085) {
        z -= 0.11;
      }
    } else {
      /*
       * ALEX: shorter structured hair.
       */
      const angle =
        u * Math.PI * 2;

      const radius =
        0.28 +
        w * 0.055;

      x =
        Math.cos(angle) *
        radius;

      y =
        0.78 +
        Math.pow(v, 0.55) *
        0.31;

      z =
        Math.sin(angle) *
        (
          0.17 +
          w * 0.045
        );

      /*
       * Slight forward crown shape.
       */
      if (z > 0.04) {
        z += 0.025;
      }
    }

    /*
     * Small organic particle variation.
     */
    x +=
      (random() - 0.5) *
      0.025;

    y +=
      (random() - 0.5) *
      0.025;

    z +=
      (random() - 0.5) *
      0.025;

    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    brightness[i] =
      THREE.MathUtils.clamp(
        identity === "sara"
          ? 0.14 + random() * 0.15
          : 0.10 + random() * 0.12,
        0,
        identity === "sara" ? 0.34 : 0.28,
      );
  }

  /*
   * Keep hair ordering stable inside the hair
   * section so SARA -> ALEX morph remains coherent.
   */
  const indices = Array.from(
    { length: count },
    (_, index) => index,
  );

  indices.sort((a, b) => {
    const ay =
      positions[a * 3 + 1];

    const by =
      positions[b * 3 + 1];

    if (Math.abs(ay - by) > 0.025) {
      return ay - by;
    }

    const az =
      positions[a * 3 + 2];

    const bz =
      positions[b * 3 + 2];

    if (Math.abs(az - bz) > 0.025) {
      return az - bz;
    }

    return (
      positions[a * 3] -
      positions[b * 3]
    );
  });

  const sortedPositions =
    new Float32Array(count * 3);

  const sortedBrightness =
    new Float32Array(count);

  indices.forEach(
    (sourceIndex, targetIndex) => {
      sortedPositions.set(
        positions.subarray(
          sourceIndex * 3,
          sourceIndex * 3 + 3,
        ),
        targetIndex * 3,
      );

      sortedBrightness[targetIndex] =
        brightness[sourceIndex];
    },
  );

  return {
    positions: sortedPositions,
    brightness: sortedBrightness,
  };
}

function mergePointData(
  body: PointData,
  hair: PointData,
): PointData {
  const bodyCount =
    body.brightness.length;

  const hairCount =
    hair.brightness.length;

  const positions =
    new Float32Array(
      (bodyCount + hairCount) * 3,
    );

  const brightness =
    new Float32Array(
      bodyCount + hairCount,
    );

  positions.set(
    body.positions,
    0,
  );

  positions.set(
    hair.positions,
    bodyCount * 3,
  );

  brightness.set(
    body.brightness,
    0,
  );

  brightness.set(
    hair.brightness,
    bodyCount,
  );

  return {
    positions,
    brightness,
  };
}

function createMorphData(
  saraScene: THREE.Object3D,
  alexScene: THREE.Object3D,
): MorphData {
  const saraSource =
    collectMeshVertices(saraScene);

  const alexSource =
    collectMeshVertices(alexScene);

  const saraBody =
    samplePointCloud(
      saraSource,
      BODY_PARTICLE_COUNT,
      1127,
      "sara",
    );

  const alexBody =
    samplePointCloud(
      alexSource,
      BODY_PARTICLE_COUNT,
      1127,
      "alex",
    );

  const saraHair =
    createProceduralHair(
      "sara",
      HAIR_PARTICLE_COUNT,
      7711,
    );

  const alexHair =
    createProceduralHair(
      "alex",
      HAIR_PARTICLE_COUNT,
      8831,
    );

  return {
    sara: mergePointData(
      saraBody,
      saraHair,
    ),
    alex: mergePointData(
      alexBody,
      alexHair,
    ),
  };
}

const vertexShader = /* glsl */ `
uniform float uTime;
uniform float uMix;
uniform float uAudio;
uniform float uListen;
uniform float uThink;
uniform float uSpeak;
uniform float uFlow;

attribute vec3 aAlex;
attribute float aSaraBrightness;
attribute float aAlexBrightness;
attribute vec4 aSeed;

varying float vBrightness;
varying float vIdentity;
varying float vFace;
varying float vFeature;

void main() {

  /*
   * Smooth SARA -> ALEX transition.
   */
  float localMix = clamp(
    (uMix - aSeed.x * 0.18) / 0.82,
    0.0,
    1.0
  );

  float morph = localMix * localMix *
    (3.0 - 2.0 * localMix);

  vec3 position = mix(
    position,
    aAlex,
    morph
  );

  /*
   * Procedural SARA hair silhouette.
   *
   * The source bust has no dedicated hair mesh, so a controlled
   * subset of the existing head/shoulder particles is pushed
   * outward to create a soft feminine hair silhouette.
   */
  float saraWeight = 1.0 - morph;

  /*
   * Give SARA a softer, more feminine silhouette:
   * narrower shoulder line and a slightly slimmer neck.
   */
  float saraShoulder =
    1.0 -
    smoothstep(
      -0.88,
      -0.32,
      position.y
    );

  float saraNeck =
    smoothstep(
      -0.42,
      0.02,
      position.y
    ) *
    (
      1.0 -
      smoothstep(
        0.02,
        0.42,
        position.y
      )
    );

  float saraBodyScale =
    1.0 -
    saraWeight *
    (
      saraShoulder * 0.075 +
      saraNeck * 0.04
    );

  position.x *= saraBodyScale;

  /*
   * Procedural SARA hair silhouette.
   *
   * Hair extends below the jaw toward the shoulders while
   * leaving the front facial surface open.
   */
  float hairTop =
    smoothstep(
      0.54,
      0.90,
      position.y
    );

  float hairSide =
    smoothstep(
      0.15,
      0.31,
      abs(position.x)
    );

  float hairLength =
    1.0 -
    smoothstep(
      -0.30,
      0.20,
      position.y
    );

  float saraHair =
    saraWeight *
    clamp(
      hairTop * 0.72 +
      hairSide * 0.48 +
      hairLength * hairSide * 0.42,
      0.0,
      1.0
    );

  float hairNoise =
    0.74 +
    0.26 *
    sin(
      aSeed.x * 41.0 +
      aSeed.y * 17.0
    );

  position.x +=
    sign(position.x + 0.0001) *
    saraHair *
    0.048 *
    hairNoise;

  position.z -=
    saraHair *
    0.055 *
    hairNoise;

  position.y -=
    hairLength *
    hairSide *
    saraWeight *
    0.025;

  /*
   * Particles travel outward during identity switching.
   */
  float transition =
    sin(morph * 3.14159265);

  vec3 flowDirection = normalize(
    aSeed.yzw + vec3(
      0.001,
      0.001,
      0.001
    )
  );

  position +=
    flowDirection *
    transition *
    uFlow;

  /*
   * Subtle living motion.
   */
  float breathing =
    sin(
      uTime * 1.15 +
      aSeed.x * 12.0
    );

  float upperBody =
    smoothstep(
      -0.85,
      0.72,
      position.y
    );

  position.y +=
    breathing *
    0.008 *
    upperBody;

  position.z +=
    breathing *
    0.006 *
    upperBody;

  /*
   * Thinking creates a neural shimmer.
   */
  vec3 thinkingMotion = vec3(
    sin(uTime * 1.8 + aSeed.x * 31.0),
    cos(uTime * 1.5 + aSeed.y * 27.0),
    sin(uTime * 2.1 + aSeed.z * 35.0)
  );

  position +=
    thinkingMotion *
    (
      0.003 +
      uThink * 0.014 +
      transition * 0.012
    );

  /*
   * Listening reacts gently to incoming audio.
   */
  position +=
    flowDirection *
    uListen *
    uAudio *
    0.018;

  /*
   * Speaking creates a subtle mouth/chest pulse.
   * The actual face shape stays stable.
   */
  float mouthDistance = distance(
    position.xy,
    vec2(
      0.0,
      0.34
    )
  );

  float mouthMask =
    1.0 -
    smoothstep(
      0.02,
      0.20,
      mouthDistance
    );

  float speechPulse =
    sin(
      uTime * 16.0 +
      aSeed.x * 8.0
    ) * 0.5 + 0.5;

  position.z +=
    mouthMask *
    uSpeak *
    uAudio *
    0.025 *
    speechPulse;

  /*
   * Identity brightness.
   */
  float brightness = mix(
    aSaraBrightness,
    aAlexBrightness,
    morph
  );

  brightness +=
    uThink *
    0.12;

  brightness +=
    uSpeak *
    uAudio *
    0.18;

  /*
   * Face region remains slightly denser/brighter.
   */
  /*
   * Facial focus.
   *
   * Use X/Y to locate the face and Z to favor the
   * front-facing facial surface. This helps preserve
   * eyes, nose and mouth depth without brightening
   * the entire head.
   */
  float faceRegion =
    exp(
      -(
        position.x * position.x * 3.2 +
        (position.y - 0.34) *
        (position.y - 0.34) * 4.2
      )
    );

  float frontDepth =
    smoothstep(
      0.02,
      0.34,
      position.z
    );

  float faceMask =
    faceRegion *
    (0.35 + frontDepth * 0.65);

  /*
   * Sharpen facial focus.
   *
   * Keep the outer head and neck restrained while giving
   * the actual front-facing facial surface more definition.
   */
  vFace = pow(
    clamp(
      faceMask,
      0.0,
      1.0
    ),
    1.55
  );

  /*
   * Facial landmarks.
   *
   * These masks do not create a new face. They selectively
   * emphasize the existing model geometry where the eyes,
   * nose bridge and mouth already exist.
   */
  float leftEye =
    exp(
      -(
        (position.x + 0.17) *
        (position.x + 0.17) * 95.0 +
        (position.y - 0.53) *
        (position.y - 0.53) * 500.0
      )
    );

  float rightEye =
    exp(
      -(
        (position.x - 0.17) *
        (position.x - 0.17) * 95.0 +
        (position.y - 0.53) *
        (position.y - 0.53) * 500.0
      )
    );

  float noseBridge =
    exp(
      -(
        position.x *
        position.x * 700.0 +
        (position.y - 0.40) *
        (position.y - 0.40) * 45.0
      )
    );

  float mouthFeature =
    exp(
      -(
        position.x *
        position.x * 55.0 +
        (position.y - 0.22) *
        (position.y - 0.22) * 500.0
      )
    );

  /*
   * Keep landmarks subtle.
   *
   * The face should read primarily from the actual model surface,
   * not from artificial glowing eye/nose/mouth shapes.
   */
  float eyeMask =
    max(leftEye, rightEye);

  float featureMask =
    max(
      eyeMask * 0.50,
      max(
        noseBridge * 0.20,
        mouthFeature * 0.25
      )
    );

  float geometryFeature =
    smoothstep(
      0.16,
      0.44,
      frontDepth
    );

  vFeature = clamp(
    max(
      featureMask * 0.45,
      geometryFeature * vFace * 0.55
    ),
    0.0,
    1.0
  );

  /*
   * Give the existing facial geometry slightly more separation
   * from the surrounding particle field.
   */
  vBrightness = clamp(
    brightness +
    vFeature * 0.10,
    0.0,
    1.0
  );

  vIdentity = morph;

  vec4 modelPosition =
    modelViewMatrix *
    vec4(position, 1.0);

  /*
   * Small but readable particle size.
   */
  /*
   * Keep the bust particles readable while making
   * facial particles finer so eyes, nose and mouth
   * resolve as detail instead of large particle rings.
   */
  float faceParticleScale =
    mix(
      1.0,
      0.58,
      vFace
    );

  float featureParticleScale =
    mix(
      1.0,
      0.68,
      vFeature
    );

  gl_PointSize =
    0.36 *
    faceParticleScale *
    featureParticleScale *
    (0.72 + vBrightness * 0.9) *
    (1.0 + uAudio * 0.35) *
    (280.0 / -modelPosition.z);

  gl_Position =
    projectionMatrix *
    modelPosition;
}
`;

const fragmentShader = /* glsl */ `
uniform vec3 uSaraColor;
uniform vec3 uAlexColor;
uniform vec3 uHighlight;

varying float vBrightness;
varying float vIdentity;
varying float vFace;
varying float vFeature;

void main() {

  vec2 centered =
    gl_PointCoord - 0.5;

  float distanceFromCenter =
    length(centered);

  float softCircle =
    1.0 -
    smoothstep(
      0.20,
      0.50,
      distanceFromCenter
    );

  /*
   * Soft luminous particles.
   */
  float alpha =
    softCircle *
    (
      0.035 +
      vBrightness * 0.27 +
      vFace * 0.09 +
      vFeature * 0.25
    );

  vec3 baseColor =
    mix(
      uSaraColor,
      uAlexColor,
      vIdentity
    );

  /*
   * Face particles receive a clean highlight.
   */
  vec3 finalColor =
    mix(
      baseColor,
      uHighlight,
      clamp(
        vFace * 0.28 +
        vFeature * 0.58 +
        pow(vBrightness, 4.0) * 0.16,
        0.0,
        0.75
      )
    );

  gl_FragColor =
    vec4(
      finalColor,
      alpha
    );
}
`;

function AvatarParticles({
  identity,
  state,
  getLevel,
}: Props) {
  const sara = useGLTF(MODELS.sara);
  const alex = useGLTF(MODELS.alex);

  const morphData = useMemo(
    () =>
      createMorphData(
        sara.scene,
        alex.scene,
      ),
    [sara, alex],
  );

  const geometry = useMemo(() => {
    const geometry =
      new THREE.BufferGeometry();

    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(
        morphData.sara.positions,
        3,
      ),
    );

    geometry.setAttribute(
      "aAlex",
      new THREE.BufferAttribute(
        morphData.alex.positions,
        3,
      ),
    );

    geometry.setAttribute(
      "aSaraBrightness",
      new THREE.BufferAttribute(
        morphData.sara.brightness,
        1,
      ),
    );

    geometry.setAttribute(
      "aAlexBrightness",
      new THREE.BufferAttribute(
        morphData.alex.brightness,
        1,
      ),
    );

    const seed = new Float32Array(
      PARTICLE_COUNT * 4,
    );

    const random = seededRandom(7719);

    for (
      let i = 0;
      i < PARTICLE_COUNT;
      i += 1
    ) {
      const direction =
        new THREE.Vector3()
          .set(
            random() * 2 - 1,
            random() * 2 - 1,
            random() * 2 - 1,
          )
          .normalize();

      seed[i * 4] = random();

      seed[i * 4 + 1] =
        direction.x;

      seed[i * 4 + 2] =
        direction.y;

      seed[i * 4 + 3] =
        direction.z;
    }

    geometry.setAttribute(
      "aSeed",
      new THREE.BufferAttribute(
        seed,
        4,
      ),
    );

    return geometry;
  }, [morphData]);

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        depthTest: true,
        blending: THREE.NormalBlending,
        uniforms: {
          uTime: {
            value: 0,
          },
          uMix: {
            value: 0,
          },
          uAudio: {
            value: 0,
          },
          uListen: {
            value: 0,
          },
          uThink: {
            value: 0,
          },
          uSpeak: {
            value: 0,
          },
          uFlow: {
            value: 0.14,
          },
          uSaraColor: {
            value: SARA_COLOR,
          },
          uAlexColor: {
            value: ALEX_COLOR,
          },
          uHighlight: {
            value: HIGHLIGHT_COLOR,
          },
        },
        vertexShader,
        fragmentShader,
      }),
    [],
  );

  const group =
    useRef<THREE.Group>(null);

  const animation =
    useRef({
      mix: 0,
      listen: 0,
      think: 0,
      speak: 0,
      audio: 0,
    });

  useFrame(
    ({ clock }, delta) => {
      const time =
        clock.elapsedTime;

      const current =
        animation.current;

      const smooth = (
        target: number,
        speed: number,
        value: number,
      ) =>
        value +
        (target - value) *
          (1 -
            Math.exp(
              -speed * delta,
            ));

      const targetMix =
        identity === "alex"
          ? 1
          : 0;

      current.mix = smooth(
        targetMix,
        2.8,
        current.mix,
      );

      current.listen = smooth(
        state === "listening"
          ? 1
          : 0,
        7,
        current.listen,
      );

      current.think = smooth(
        state === "thinking"
          ? 1
          : 0,
        5,
        current.think,
      );

      current.speak = smooth(
        state === "speaking"
          ? 1
          : 0,
        7,
        current.speak,
      );

      const suppliedLevel =
        getLevel?.() ?? 0;

      const fallbackLevel =
        state === "speaking"
          ? (
              Math.abs(
                Math.sin(
                  time * 6.7,
                ),
              ) *
              Math.abs(
                Math.sin(
                  time * 2.0,
                ),
              ) *
              0.75
            )
          : 0;

      const targetAudio =
        Math.max(
          0,
          Math.min(
            1,
            suppliedLevel ||
              fallbackLevel,
          ),
        );

      current.audio = smooth(
        targetAudio,
        16,
        current.audio,
      );

      material.uniforms.uTime.value =
        time;

      material.uniforms.uMix.value =
        current.mix;

      material.uniforms.uAudio.value =
        current.audio;

      material.uniforms.uListen.value =
        current.listen;

      material.uniforms.uThink.value =
        current.think;

      material.uniforms.uSpeak.value =
        current.speak;

      if (group.current) {
        group.current.rotation.y =
          Math.sin(time * 0.38) *
          0.035;

        group.current.position.y =
          Math.sin(time * 0.8) *
          0.008;
      }
    },
  );

  return (
    <group ref={group}>
      <points
        geometry={geometry}
        material={material}
        frustumCulled={false}
      />
    </group>
  );
}

function AmbientParticles({
  state,
}: {
  state: AvatarState;
}) {
  const points =
    useRef<THREE.Points>(null);

  const geometry = useMemo(() => {
    const count = 420;

    const positions =
      new Float32Array(
        count * 3,
      );

    const random =
      seededRandom(9021);

    for (
      let i = 0;
      i < count;
      i += 1
    ) {
      const radius =
        1.15 +
        random() * 0.9;

      const angle =
        random() *
        Math.PI *
        2;

      positions[i * 3] =
        Math.cos(angle) *
        radius *
        0.75;

      positions[i * 3 + 1] =
        (random() - 0.5) *
        2.2;

      positions[i * 3 + 2] =
        (random() - 0.5) *
        0.55;
    }

    const geometry =
      new THREE.BufferGeometry();

    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(
        positions,
        3,
      ),
    );

    return geometry;
  }, []);

  useFrame(({ clock }) => {
    if (!points.current) {
      return;
    }

    points.current.rotation.y =
      clock.elapsedTime * 0.025;

    points.current.rotation.z =
      Math.sin(
        clock.elapsedTime * 0.2,
      ) * 0.02;

    const material =
      points.current.material as THREE.PointsMaterial;

    material.opacity =
      state === "thinking"
        ? 0.24
        : state === "listening"
          ? 0.16
          : 0.08;
  });

  return (
    <points
      ref={points}
      geometry={geometry}
      frustumCulled={false}
    >
      <pointsMaterial
        size={0.012}
        color="#8bbdff"
        transparent
        opacity={0.08}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

function Scene(props: Props) {
  return (
    <>
      <AmbientParticles
        state={props.state}
      />

      <AvatarParticles
        {...props}
      />
    </>
  );
}

export default function ProzynAvatar(
  props: Props,
) {
  return (
    <Canvas
      dpr={[1, 2]}
      camera={{
        position: [0, 0, 6.4],
        fov: 28,
      }}
      gl={{
        antialias: true,
        alpha: true,
        powerPreference:
          "high-performance",
      }}
      onCreated={({ gl }) => {
        gl.setClearColor(
          0x000000,
          0,
        );
      }}
      style={{
        width: "100%",
        height: "100%",
        background:
          "transparent",
      }}
    >
      <Suspense fallback={null}>
        <Scene {...props} />
      </Suspense>
    </Canvas>
  );
}

useGLTF.preload(MODELS.sara);
useGLTF.preload(MODELS.alex);