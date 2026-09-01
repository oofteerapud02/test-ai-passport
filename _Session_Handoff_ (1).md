# บริบทโปรเจกต์: BVH → VRMA Retargeting

## เป้าหมาย
แปลง motion capture (BVH) → VRM Animation (.vrma)
retarget world rotations → VRM normalized humanoid local rotations

## โครงสร้างไฟล์
- converter/retarget.py  — build_tracks() หัวใจการแปลง
- converter/skeleton.py  — REST, BONES, PARENT, OFFSET, _EXPLICIT_DIR, target_rest_dirs()
- converter/motion.py    — dataclass Motion
- converter/cli.py       — entry point
- config.py              — TARGET_FRAMES

## อาการ
ตัวยืนถูกตำแหน่ง แต่ **แขนไขว้หุบเข้าใน / เข่าบิดเข้าหากัน**

## สมการหลัก (ยืนยันแล้วว่าถูก)
T_b(t) = W_b(t) · A_b   โดย A_b · t_b = d_b
local_b = T_parent(t)⁻¹ · T_b(t)
→ ยุบเป็น local = A_p⁻¹ · (W_p⁻¹ W_c) · A_c

## แก้ไปแล้ว (ยังไม่ได้รันเทส)
1. retarget.py — _rot_between() เดิมให้ minimal-arc → twist ไม่ถูกกำหนด
   แก้: เขียน _basis() + _align() สร้าง orthonormal frame
   ด้วย reference vector เดียวกันทุกกระดูก (_REF_PRIMARY = +Z, fallback +Y)
2. skeleton.py — leaf fallback ±X → เพิ่ม _LEAF_DIR
3. skeleton.py — "ลูกตัวแรกชนะ" → เพิ่ม _PREFERRED_CHILD + children{} เก็บลูกทุกตัว
   + fallback เฉลี่ยลูกหลายตัว แทนหยิบตัวแรก
4. retarget.py — `if motion.space=="src" and up=="z"` → `if up=="z"`
5. retarget.py — rest_dir ไม่ครบ → ปิด rest_correct ทั้งชุด (ไม่ทำครึ่งๆ)
6. retarget.py — raise ถ้าจำนวนเฟรมของ track ไม่เท่า hips
7. skeleton.py — _norm() วอร์นเมื่อเจอเวกเตอร์ศูนย์
8. เพิ่ม _debug_dump() (env RETARGET_DEBUG=1) พิมพ์ euler ทุก bone
   + ตรวจสมมาตรซ้าย-ขวา (X เท่ากัน, Y/Z กลับเครื่องหมาย)

## ตรวจ skeleton.py แล้ว — ผ่านหมด ไม่ใช่ต้นเหตุ
- _EXPLICIT_DIR ครอบคลุม: hips(0,1,0), head(0,1,0), hands(±1,0,0),
  toes(0,0,1), jaw, eyes
- leaf ที่เหลือ = ปลายนิ้ว → fallback ±X ถูกอยู่แล้ว (dx = sign*_FINGER_LEN)
- REST สะอาด: upperLeg(0,−0.4,0), foot(0,−0.06,0.1)
- chest ปลอดภัย: ("chest","spine",(0,0.12,0)) + ("upperChest","chest",(0,0.12,0))
  ประกาศติดกัน → chest ได้ +Y เสมอ
- ยังไม่ยืนยัน: ลำดับลูกของ upperChest (neck ต้องมาก่อน leftShoulder/rightShoulder)

## 🎯 สมมติฐานหลักตอนนี้ — motion.py: rest_dir เป็น optional
```python
rest_dir: Dict[str, np.ndarray] = field(default_factory=dict)  # ดีฟอลต์ = ว่าง!
space: str = "src"   # ดีฟอลต์
up: str = "y"
world_rot: Dict[str, (N,4) xyzw]
```
ถ้า loader ไม่เซ็ต rest_dir → (ตามแพตช์ข้อ 5) A_b = I ทุกกระดูก
→ ไม่ retarget เลย → world rotation ดิบยัดใส่ VRM
→ อาการตรงเป๊ะ: ตัวยืนถูก แต่แขน/ขาบิดหุบ

## STEP ถัดไป (ทำก่อนอย่างอื่น)
ใส่ก่อนเรียก build_tracks() ใน cli.py:
```python
print(f"[chk] rest_dir: {len(motion.rest_dir)} bones | space={motion.space} | up={motion.up}")
if motion.rest_dir:
    for b in ("hips","leftUpperLeg","leftHand","upperChest"):
        print(f"      {b:<14}{motion.rest_dir.get(b)}")
```
อ่านผล:
| ผลลัพธ์ | แปลว่า |
|---|---|
| `0 bones` | 🎯 เจอตัวการ — rest_correct ไม่เคยทำงาน |
| hips = −Y | source ใช้ขาเป็นลูกตัวแรก → ต้อง pin +Y |
| hips = +Y ครบทุก bone | ปัญหาอยู่ที่ _align()/ลำดับคูณ quaternion |

## จุดรองที่ต้องเช็ค
- space ดีฟอลต์ "src" — ถ้า loader ไม่เซ็ตเป็น "vrm" หลังแปลง อาจแปลงแกนซ้ำ 2 รอบ
- world_rot เป็น xyzw (ตรงกับ scipy) แต่ glTF/VRMA บางจุดใช้ wxyz → เช็คตอน export

## วิธีรัน
RETARGET_DEBUG=1 python -m converter.cli "hand wave.bvh"
PowerShell: $env:RETARGET_DEBUG=1; python -m converter.cli "hand wave.bvh"
เกณฑ์: worst asymmetry ≤ 25° = ผ่าน
ถ้าขายังบิด → ลองเปลี่ยน _REF_PRIMARY จาก +Z เป็น +X

## สิ่งที่ต้องแนบในแชทใหม่
1. ไฟล์ที่สร้าง Motion (bvh.py / loader.py / parser.py) — ตัวที่ cli.py
   เรียกก่อน build_tracks โดยเฉพาะบรรทัดที่เซ็ต rest_dir=
2. ผลรันสคริปต์ [chk] ข้างบน
3. (ถ้ามี) skeleton.py บรรทัด 1–21 เพื่อยืนยันลำดับลูกของ upperChest