package com.hcl.app1.service;

import java.util.HashMap;
import java.util.Map;
import org.springframework.stereotype.Service;

@Service
public class UserService {

    public Map<String, Object> getUserById(String id) {
        Map<String, Object> response = new HashMap<>();
        response.put("id", id);
        response.put("name", "John Doe");
        response.put("status", "ACTIVE");
        return response;
    }

    public Map<String, Object> createUser(Map<String, Object> payload) {
        Map<String, Object> response = new HashMap<>(payload);
        response.put("message", "User created successfully");
        return response;
    }

    public Map<String, Object> updateUser(String id, Map<String, Object> payload) {
        Map<String, Object> response = new HashMap<>(payload);
        response.put("id", id);
        response.put("message", "User updated successfully");
        return response;
    }

    public Map<String, Object> deleteUser(String id) {
        Map<String, Object> response = new HashMap<>();
        response.put("id", id);
        response.put("message", "User deleted successfully");
        return response;
    }
}
