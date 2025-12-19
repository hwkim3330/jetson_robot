/**
 * Robot Navigation Library
 * Lightweight JS-based navigation algorithms for SDV robot
 * - Frontier Exploration (auto SLAM)
 * - Goal Navigation with obstacle avoidance
 * - Path finding helpers
 */

const Nav = (function() {
    'use strict';

    // Configuration
    const config = {
        linearSpeed: 0.18,
        angularSpeed: 0.8,
        obstacleDistance: 0.4,
        sideObstacleDistance: 0.3,
        goalReachThreshold: 0.15,
        frontierMinDist: 0.5,
        frontierMaxDist: 10
    };

    // State
    let navInterval = null;
    let exploreInterval = null;
    let frontierGoal = null;
    let lastFrontierSearch = 0;

    // Stuck detection
    let stuckCount = 0;
    let lastPose = {x: 0, y: 0};
    let recoveryMode = false;
    let recoveryStart = 0;

    /**
     * Find frontier cells (free cells adjacent to unknown)
     * @param {Array} mapData - OccupancyGrid data array
     * @param {Object} mapInfo - Map metadata (width, height, resolution, origin)
     * @returns {Array} Array of {x, y} frontier points in world coordinates
     */
    function findFrontiers(mapData, mapInfo) {
        if(!mapData || !mapInfo) return [];

        const w = mapInfo.width;
        const h = mapInfo.height;
        const res = mapInfo.resolution;
        const ox = mapInfo.origin.position.x;
        const oy = mapInfo.origin.position.y;

        const frontiers = [];

        // Sample every 4th cell for speed
        for(let y = 2; y < h - 2; y += 4) {
            for(let x = 2; x < w - 2; x += 4) {
                const idx = y * w + x;
                if(mapData[idx] !== 0) continue; // Not free

                // Check if adjacent to unknown
                let hasUnknown = false;
                for(let dy = -2; dy <= 2; dy++) {
                    for(let dx = -2; dx <= 2; dx++) {
                        const ni = (y + dy) * w + (x + dx);
                        if(ni >= 0 && ni < mapData.length && mapData[ni] === -1) {
                            hasUnknown = true;
                            break;
                        }
                    }
                    if(hasUnknown) break;
                }

                if(hasUnknown) {
                    const wx = ox + x * res;
                    const wy = oy + y * res;
                    frontiers.push({x: wx, y: wy});
                }
            }
        }

        return frontiers;
    }

    /**
     * Find nearest frontier to robot
     * @param {Array} mapData - OccupancyGrid data
     * @param {Object} mapInfo - Map metadata
     * @param {Object} robotPose - Robot position {x, y, theta}
     * @returns {Object|null} Nearest frontier {x, y} or null
     */
    function findNearestFrontier(mapData, mapInfo, robotPose) {
        const frontiers = findFrontiers(mapData, mapInfo);
        if(frontiers.length === 0) return null;

        let nearest = null;
        let minDist = Infinity;

        for(const f of frontiers) {
            const dx = f.x - robotPose.x;
            const dy = f.y - robotPose.y;
            const dist = Math.sqrt(dx*dx + dy*dy);

            if(dist < config.frontierMinDist || dist > config.frontierMaxDist) continue;

            if(dist < minDist) {
                minDist = dist;
                nearest = f;
            }
        }

        return nearest;
    }

    /**
     * Check obstacles from lidar points
     * @param {Array} pts - Lidar points [{x, y, d}, ...]
     * @param {number} obstDist - Obstacle detection distance
     * @returns {Object} {front: bool, left: bool, right: bool} - true if clear
     */
    function checkObstacles(pts, obstDist = config.obstacleDistance) {
        let frontClear = true;
        let leftClear = true;
        let rightClear = true;

        for(const p of pts) {
            // Front sector
            if(p.x > 0 && Math.abs(p.y) < 0.25 && p.d < obstDist) frontClear = false;
            // Left sector
            if(p.x > 0 && p.y > 0.1 && p.d < obstDist * 0.7) leftClear = false;
            // Right sector
            if(p.x > 0 && p.y < -0.1 && p.d < obstDist * 0.7) rightClear = false;
        }

        return { front: frontClear, left: leftClear, right: rightClear };
    }

    /**
     * Calculate velocity command to reach goal with obstacle avoidance
     * @param {Object} robotPose - Current robot pose {x, y, theta}
     * @param {Object} goalPose - Goal position {x, y}
     * @param {Array} pts - Lidar points
     * @returns {Object} {lin, ang, arrived} velocity command and status
     */
    function computeNavCmd(robotPose, goalPose, pts) {
        if(!goalPose) return {lin: 0, ang: 0, arrived: false};

        const dx = goalPose.x - robotPose.x;
        const dy = goalPose.y - robotPose.y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        const goalAngle = Math.atan2(dy, dx);

        // Normalize angle difference
        let angleDiff = goalAngle - robotPose.theta;
        while(angleDiff > Math.PI) angleDiff -= 2*Math.PI;
        while(angleDiff < -Math.PI) angleDiff += 2*Math.PI;

        // Check if arrived
        if(dist < config.goalReachThreshold) {
            return {lin: 0, ang: 0, arrived: true};
        }

        const obs = checkObstacles(pts);
        let lin = 0, ang = 0;

        if(!obs.front) {
            // Obstacle ahead - rotate to avoid
            ang = obs.left ? config.angularSpeed : (obs.right ? -config.angularSpeed : config.angularSpeed);
            lin = 0;
        } else if(Math.abs(angleDiff) > 0.3) {
            // Rotate toward goal
            ang = angleDiff > 0 ? 0.6 : -0.6;
            lin = 0.05;
        } else {
            // Go forward
            lin = Math.min(config.linearSpeed, dist * 0.5);
            ang = angleDiff * 1.0;
        }

        return {lin, ang, arrived: false};
    }

    /**
     * Compute exploration velocity command (frontier-based)
     * @param {Object} robotPose - Current pose
     * @param {Array} mapData - Map occupancy data
     * @param {Object} mapInfo - Map info
     * @param {Array} pts - Lidar points
     * @returns {Object} {lin, ang, goal, complete, recovering}
     */
    function computeExploreCmd(robotPose, mapData, mapInfo, pts) {
        const now = Date.now();

        // Stuck detection - check if robot hasn't moved
        const moved = Math.sqrt(
            Math.pow(robotPose.x - lastPose.x, 2) +
            Math.pow(robotPose.y - lastPose.y, 2)
        );

        if(moved < 0.02) {
            stuckCount++;
        } else {
            stuckCount = Math.max(0, stuckCount - 2);
            lastPose = {x: robotPose.x, y: robotPose.y};
        }

        // Enter recovery mode if stuck for too long (15 iterations = ~1.5s)
        if(stuckCount > 15 && !recoveryMode) {
            recoveryMode = true;
            recoveryStart = now;
            frontierGoal = null; // Clear current goal
        }

        // Recovery mode - back up and spin
        if(recoveryMode) {
            const elapsed = now - recoveryStart;
            if(elapsed < 1500) {
                // Phase 1: Back up
                return {lin: -0.12, ang: 0, goal: null, complete: false, recovering: true};
            } else if(elapsed < 3000) {
                // Phase 2: Spin in place
                return {lin: 0, ang: 1.0, goal: null, complete: false, recovering: true};
            } else {
                // Recovery complete
                recoveryMode = false;
                stuckCount = 0;
                lastFrontierSearch = 0;
            }
        }

        // Find new frontier every 3 seconds or when reached
        const needNewGoal = !frontierGoal ||
            (now - lastFrontierSearch > 3000) ||
            (frontierGoal && Math.sqrt(
                Math.pow(frontierGoal.x - robotPose.x, 2) +
                Math.pow(frontierGoal.y - robotPose.y, 2)) < 0.3);

        if(needNewGoal) {
            lastFrontierSearch = now;
            frontierGoal = findNearestFrontier(mapData, mapInfo, robotPose);
        }

        const obs = checkObstacles(pts);
        let lin = 0, ang = 0;

        // Get minimum distances for smarter avoidance
        let minFront = 10, minLeft = 10, minRight = 10;
        for(const p of pts) {
            if(p.rx > 0 && Math.abs(p.ry) < 0.2 && p.d < minFront) minFront = p.d;
            if(p.rx > -0.1 && p.ry > 0.15 && p.d < minLeft) minLeft = p.d;
            if(p.rx > -0.1 && p.ry < -0.15 && p.d < minRight) minRight = p.d;
        }

        if(frontierGoal) {
            const dx = frontierGoal.x - robotPose.x;
            const dy = frontierGoal.y - robotPose.y;
            const goalAngle = Math.atan2(dy, dx);
            let angleDiff = goalAngle - robotPose.theta;
            while(angleDiff > Math.PI) angleDiff -= 2*Math.PI;
            while(angleDiff < -Math.PI) angleDiff += 2*Math.PI;

            if(!obs.front) {
                // Obstacle ahead - turn toward more open side
                ang = minLeft > minRight ? config.angularSpeed : -config.angularSpeed;
                // Slight backup if very close
                lin = minFront < 0.25 ? -0.08 : 0;
            } else if(Math.abs(angleDiff) > 0.4) {
                ang = angleDiff > 0 ? 0.6 : -0.6;
                lin = 0.05;
            } else {
                lin = config.linearSpeed;
                ang = angleDiff * 0.8;
                // Wall following tendency
                if(!obs.left) ang -= 0.15;
                if(!obs.right) ang += 0.15;
            }
        } else {
            // No frontier - wander or complete
            if(!obs.front) {
                ang = minLeft > minRight ? config.angularSpeed : -config.angularSpeed;
            } else {
                lin = 0.12;
                ang = (minLeft - minRight) * 0.5;  // Steer away from closer wall
            }
        }

        return {
            lin, ang,
            goal: frontierGoal,
            complete: !frontierGoal && findFrontiers(mapData, mapInfo).length === 0,
            recovering: false
        };
    }

    /**
     * Simple A* pathfinding on occupancy grid
     * @param {Array} mapData - Occupancy grid data
     * @param {Object} mapInfo - Map info
     * @param {Object} start - Start position {x, y} in world coords
     * @param {Object} goal - Goal position {x, y} in world coords
     * @returns {Array} Path as [{x, y}, ...] in world coords
     */
    function findPath(mapData, mapInfo, start, goal) {
        if(!mapData || !mapInfo) return [];

        const w = mapInfo.width;
        const h = mapInfo.height;
        const res = mapInfo.resolution;
        const ox = mapInfo.origin.position.x;
        const oy = mapInfo.origin.position.y;

        // Convert world to grid
        const toGrid = (wx, wy) => ({
            gx: Math.floor((wx - ox) / res),
            gy: Math.floor((wy - oy) / res)
        });
        const toWorld = (gx, gy) => ({
            x: ox + gx * res + res/2,
            y: oy + gy * res + res/2
        });

        const sg = toGrid(start.x, start.y);
        const eg = toGrid(goal.x, goal.y);

        // Check bounds
        if(sg.gx < 0 || sg.gx >= w || sg.gy < 0 || sg.gy >= h) return [];
        if(eg.gx < 0 || eg.gx >= w || eg.gy < 0 || eg.gy >= h) return [];

        // A* algorithm
        const openSet = new Set();
        const cameFrom = new Map();
        const gScore = new Map();
        const fScore = new Map();

        const key = (x, y) => `${x},${y}`;
        const heuristic = (x1, y1, x2, y2) => Math.abs(x1-x2) + Math.abs(y1-y2);

        const startKey = key(sg.gx, sg.gy);
        openSet.add(startKey);
        gScore.set(startKey, 0);
        fScore.set(startKey, heuristic(sg.gx, sg.gy, eg.gx, eg.gy));

        const dirs = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]];

        while(openSet.size > 0) {
            // Find lowest fScore
            let current = null;
            let lowestF = Infinity;
            for(const k of openSet) {
                const f = fScore.get(k) || Infinity;
                if(f < lowestF) { lowestF = f; current = k; }
            }

            if(!current) break;

            const [cx, cy] = current.split(',').map(Number);

            // Reached goal?
            if(cx === eg.gx && cy === eg.gy) {
                // Reconstruct path
                const path = [];
                let curr = current;
                while(curr) {
                    const [px, py] = curr.split(',').map(Number);
                    path.unshift(toWorld(px, py));
                    curr = cameFrom.get(curr);
                }
                return path;
            }

            openSet.delete(current);

            // Check neighbors
            for(const [dx, dy] of dirs) {
                const nx = cx + dx;
                const ny = cy + dy;

                if(nx < 0 || nx >= w || ny < 0 || ny >= h) continue;

                const idx = ny * w + nx;
                if(mapData[idx] > 50) continue; // Obstacle

                const neighborKey = key(nx, ny);
                const tentativeG = (gScore.get(current) || 0) + (dx && dy ? 1.414 : 1);

                if(tentativeG < (gScore.get(neighborKey) || Infinity)) {
                    cameFrom.set(neighborKey, current);
                    gScore.set(neighborKey, tentativeG);
                    fScore.set(neighborKey, tentativeG + heuristic(nx, ny, eg.gx, eg.gy));
                    openSet.add(neighborKey);
                }
            }

            // Limit iterations
            if(gScore.size > 10000) break;
        }

        return []; // No path found
    }

    /**
     * Reset exploration state
     */
    function resetExplore() {
        frontierGoal = null;
        lastFrontierSearch = 0;
        stuckCount = 0;
        recoveryMode = false;
        lastPose = {x: 0, y: 0};
    }

    // Public API
    return {
        config,
        findFrontiers,
        findNearestFrontier,
        checkObstacles,
        computeNavCmd,
        computeExploreCmd,
        findPath,
        resetExplore
    };
})();

// Export for module systems
if(typeof module !== 'undefined' && module.exports) {
    module.exports = Nav;
}
