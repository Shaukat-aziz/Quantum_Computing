import numpy as np

# Parameters
a = 1.0
b = 1.0
rx = 2.0
ry = b


def is_in_stadium(x, y):
    if abs(x) <= a:
        return abs(y) <= b
    x_shifted = abs(x) - a
    return (x_shifted/rx)**2 + (y/ry)**2 <= 1


def get_normal(x, y):
    tol = 1e-9
    sx = np.sign(x) if x != 0 else 1
    if abs(x) <= a + tol and abs(y) >= b - tol:
        return np.array([0.0, np.sign(y)])
    if abs(y) <= b + tol and abs(x) >= a - tol:
        return np.array([np.sign(x), 0.0])
    if abs(x) > a - tol:
        x_shifted = x - sx * a
        return np.array([x_shifted/(rx**2), y/(ry**2)])
    return np.array([np.sign(x), np.sign(y)])


def reflect_velocity(v, normal):
    tol = 1e-9
    norm = np.linalg.norm(normal)
    if norm == 0:
        return v.copy()
    n = normal / norm
    if abs(n[0]) < tol:
        n[0] = 0.0
    if abs(n[1]) < tol:
        n[1] = 0.0
    if n[0] == 0.0 and n[1] != 0.0:
        v_ref = np.array([0.0 if abs(v[0]) < 1e-12 else v[0], -v[1]])
    elif n[1] == 0.0 and n[0] != 0.0:
        v_ref = np.array([-v[0], 0.0 if abs(v[1]) < 1e-12 else v[1]])
    else:
        v_ref = v - 2 * np.dot(v, n) * n
    v_ref[np.abs(v_ref) < 1e-12] = 0.0
    return v_ref


def simulate_trajectory_debug(x0, y0, vx0, vy0, dt=0.01, steps=1000, show_progress=False):
    x_list, y_list = [x0], [y0]
    vx, vy = float(vx0), float(vy0)
    bounce_count = 0
    logs = []

    for i in range(steps):
        x_prev, y_prev = x_list[-1], y_list[-1]
        x_new = x_prev + vx * dt
        y_new = y_prev + vy * dt

        if not is_in_stadium(x_new, y_new):
            # locate collision
            t_lo, t_hi = 0.0, 1.0
            def interp(t):
                return x_prev + (x_new - x_prev) * t, y_prev + (y_new - y_prev) * t
            for _ in range(80):
                tm = 0.5 * (t_lo + t_hi)
                xm, ym = interp(tm)
                if is_in_stadium(xm, ym):
                    t_lo = tm
                else:
                    t_hi = tm
                if t_hi - t_lo < 1e-12:
                    break
            t_coll = 0.5 * (t_lo + t_hi)
            x_coll, y_coll = interp(t_coll)

            v_pre = np.array([vx, vy])
            speed_pre = np.hypot(v_pre[0], v_pre[1])
            normal = get_normal(x_coll, y_coll)
            v_ref = reflect_velocity(np.array([vx, vy]), normal)
            speed_post = np.hypot(v_ref[0], v_ref[1])

            # compute incidence/reflection dot with normal
            nnorm = np.linalg.norm(normal)
            nvec = normal / nnorm if nnorm>0 else np.array([0.0,0.0])
            inc = np.dot(v_pre, nvec)
            ref = np.dot(v_ref, nvec)

            logs.append((bounce_count+1, i, x_coll, y_coll, speed_pre, speed_post, inc, ref))

            # fallback analytic reflection if necessary
            if speed_post < 1e-12 and nnorm>0:
                v_ref = v_pre - 2 * np.dot(v_pre, nvec) * nvec
                speed_post = np.hypot(v_ref[0], v_ref[1])

            # renormalize
            if speed_post>0 and speed_pre>0:
                v_ref = v_ref * (speed_pre / speed_post)

            vx, vy = float(v_ref[0]), float(v_ref[1])
            bounce_count += 1
            x_list.append(x_coll); y_list.append(y_coll)

            remaining = (1.0 - t_coll)
            x_new = x_coll + vx * dt * remaining
            y_new = y_coll + vy * dt * remaining

            if not is_in_stadium(x_new, y_new):
                x_new, y_new = x_coll, y_coll

            # nudge
            if abs(x_new - x_coll) < 1e-12 and abs(y_new - y_coll) < 1e-12:
                vn = np.hypot(vx, vy)
                if vn > 1e-16:
                    x_new += (vx / vn) * 1e-9
                    y_new += (vy / vn) * 1e-9
                else:
                    vpre_norm = np.hypot(v_pre[0], v_pre[1])
                    if vpre_norm>0:
                        x_new += (v_pre[0]/vpre_norm)*1e-9
                        y_new += (v_pre[1]/vpre_norm)*1e-9

        x_list.append(x_new); y_list.append(y_new)

        # stop if too far
        if abs(x_new) > 10*a or abs(y_new) > 10*b:
            break

    return np.array(x_list), np.array(y_list), vx, vy, logs


if __name__ == '__main__':
    tests = [
        (0,0,0.1,0.13,0.19,1500),
        (0.1,0,0,1,0.01,200),
    ]

    for idx, (x0,y0,vx0,vy0,dt,steps) in enumerate(tests,1):
        print(f"\nTest {idx}: x0={x0}, y0={y0}, vx0={vx0}, vy0={vy0}, dt={dt}, steps={steps}")
        X,Y,vx_final,vy_final,logs = simulate_trajectory_debug(x0,y0,vx0,vy0,dt=dt,steps=steps,show_progress=True)
        print(f"Final speed: {np.hypot(vx_final,vy_final):.6g}")
        print(f"Number of points: {len(X)}, bounces logged: {len(logs)}")
        for log in logs[:10]:
            bounce,i,xcoll,ycoll,s_pre,s_post,inc,ref = log
            print(f"Bounce {bounce} at iter {i}: pos=({xcoll:.6g},{ycoll:.6g}), speed_pre={s_pre:.6g}, speed_post={s_post:.6g}, v·n pre={inc:.6g}, v·n post={ref:.6g}")
        if len(logs)==0:
            print("No bounces detected")
        print("---")
